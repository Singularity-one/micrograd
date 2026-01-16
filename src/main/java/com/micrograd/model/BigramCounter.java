package com.micrograd.model;

import com.micrograd.data.Vocabulary;
import com.micrograd.tensor.Tensor;
import com.micrograd.tensor.TensorOps;
import com.micrograd.util.RandomUtils;

import java.util.Random;

/**
 * Bigram 語言模型 - 計數方法
 * 透過統計 bigram 出現次數來建立機率分佈
 */
public class BigramCounter implements LanguageModel {

    private final Vocabulary vocab;
    private final Tensor counts;    // 27×27 計數矩陣
    private Tensor probs;           // 27×27 機率矩陣
    private final double smoothing; // Laplace smoothing

    public BigramCounter(Vocabulary vocab) {
        this(vocab, 1.0);  // 預設 smoothing = 1
    }

    public BigramCounter(Vocabulary vocab, double smoothing) {
        this.vocab = vocab;
        this.counts = Tensor.zeros(vocab.size(), vocab.size());
        this.smoothing = smoothing;
    }

    @Override
    public void train(int[][] bigrams) {
        int[] inputs = bigrams[0];
        int[] targets = bigrams[1];

        // 統計計數
        for (int i = 0; i < inputs.length; i++) {
            int ix1 = inputs[i];
            int ix2 = targets[i];
            counts.increment(ix1, ix2, 1.0);
        }

        // 加上 smoothing 並正規化
        Tensor smoothedCounts = counts.add(smoothing);
        probs = TensorOps.normalizeRows(smoothedCounts);
    }

    @Override
    public double loss(int[][] bigrams) {
        int[] inputs = bigrams[0];
        int[] targets = bigrams[1];

        double totalLoss = 0;
        int n = inputs.length;

        for (int i = 0; i < n; i++) {
            int ix1 = inputs[i];
            int ix2 = targets[i];
            double prob = probs.get(ix1, ix2);
            totalLoss += -Math.log(prob);
        }

        return totalLoss / n;
    }

    @Override
    public String sample(Random rng) {
        StringBuilder result = new StringBuilder();
        int idx = 0;  // 從特殊符號 '.' 開始

        while (true) {
            // 取得當前字元的機率分佈
            Tensor prob = probs.getRow(idx);

            // 根據機率採樣下一個字元
            idx = RandomUtils.multinomial(prob, rng);

            // 如果是結束符號，停止
            if (idx == 0) {
                break;
            }

            // 加入結果
            result.append(vocab.decode(idx));
        }

        return result.toString();
    }

    @Override
    public String getName() {
        return "BigramCounter (smoothing=" + smoothing + ")";
    }

    /**
     * 取得計數矩陣
     */
    public Tensor getCounts() {
        return counts;
    }

    /**
     * 取得機率矩陣
     */
    public Tensor getProbs() {
        return probs;
    }

    /**
     * 印出某個字元之後最可能的字元
     */
    public void printTopNext(char c, int topK) {
        int idx = vocab.encode(c);
        Tensor prob = probs.getRow(idx);

        System.out.printf("'%c' 之後最可能的 %d 個字元:\n", c, topK);

        // 簡單的 top-k 實作
        boolean[] used = new boolean[vocab.size()];
        for (int k = 0; k < topK; k++) {
            double maxProb = -1;
            int maxIdx = -1;
            for (int j = 0; j < vocab.size(); j++) {
                if (!used[j] && prob.get(0, j) > maxProb) {
                    maxProb = prob.get(0, j);
                    maxIdx = j;
                }
            }
            if (maxIdx >= 0) {
                used[maxIdx] = true;
                char nextChar = vocab.decode(maxIdx);
                String display = (nextChar == '.') ? "END" : String.valueOf(nextChar);
                System.out.printf("  %s: %.4f (%.2f%%)\n", display, maxProb, maxProb * 100);
            }
        }
    }

    /**
     * 檢查機率分佈是否正規化
     */
    public boolean checkNormalization() {
        for (int i = 0; i < vocab.size(); i++) {
            double rowSum = probs.getRow(i).sum();
            if (Math.abs(rowSum - 1.0) > 1e-6) {
                System.err.printf("Row %d sum = %.6f (expected 1.0)\n", i, rowSum);
                return false;
            }
        }
        return true;
    }
}