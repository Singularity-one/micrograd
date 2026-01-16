package com.micrograd.model;

import com.micrograd.data.Vocabulary;
import com.micrograd.tensor.Tensor;
import com.micrograd.tensor.TensorOps;
import com.micrograd.util.RandomUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Bigram 語言模型 - 神經網路方法
 * 使用梯度下降學習機率分佈
 */
public class BigramNeuralNet implements LanguageModel {

    private final Vocabulary vocab;
    private final Tensor W;           // 權重矩陣 27×27
    private final double learningRate;
    private final int epochs;
    private final Random initRng;

    public BigramNeuralNet(Vocabulary vocab, double learningRate, int epochs, Random rng) {
        this.vocab = vocab;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.initRng = rng;

        // 初始化權重（隨機）
        this.W = Tensor.randn(vocab.size(), vocab.size(), rng);
    }

    public BigramNeuralNet(Vocabulary vocab) {
        this(vocab, 50.0, 100, new Random(42));
    }

    @Override
    public void train(int[][] bigrams) {
        int[] inputs = bigrams[0];
        int[] targets = bigrams[1];
        int n = inputs.length;

        System.out.println("開始訓練神經網路...");
        System.out.println("樣本數量: " + n);
        System.out.println("學習率: " + learningRate);
        System.out.println("訓練輪數: " + epochs);

        for (int epoch = 0; epoch < epochs; epoch++) {
            // ===== 前向傳播 =====

            // 1. One-hot 編碼輸入
            Tensor xenc = TensorOps.oneHotBatch(inputs, vocab.size());  // n×27

            // 2. 計算 logits = xenc @ W
            Tensor logits = xenc.matmul(W);  // n×27

            // 3. Softmax 得到機率
            Tensor probs = TensorOps.softmax(logits);  // n×27

            // 4. 計算損失
            double loss = TensorOps.negativeLogLikelihood(probs, targets);

            // ===== 反向傳播 =====

            // 計算梯度: dL/dlogits = probs - one_hot(targets)
            Tensor dlogits = probs.copy();
            for (int i = 0; i < n; i++) {
                int targetIdx = targets[i];
                dlogits.set(i, targetIdx, dlogits.get(i, targetIdx) - 1.0);
            }
            // 平均梯度
            dlogits = dlogits.div(n);

            // dL/dW = xenc.T @ dlogits
            Tensor dW = xenc.transpose().matmul(dlogits);

            // ===== 更新權重 =====
            for (int i = 0; i < vocab.size(); i++) {
                for (int j = 0; j < vocab.size(); j++) {
                    double oldW = W.get(i, j);
                    double grad = dW.get(i, j);
                    W.set(i, j, oldW - learningRate * grad);
                }
            }

            // 印出進度
            if (epoch % 10 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %3d | Loss: %.4f\n", epoch, loss);
            }
        }
    }

    @Override
    public double loss(int[][] bigrams) {
        int[] inputs = bigrams[0];
        int[] targets = bigrams[1];

        Tensor xenc = TensorOps.oneHotBatch(inputs, vocab.size());
        Tensor logits = xenc.matmul(W);
        Tensor probs = TensorOps.softmax(logits);

        return TensorOps.negativeLogLikelihood(probs, targets);
    }

    @Override
    public String sample(Random rng) {
        StringBuilder result = new StringBuilder();
        int idx = 0;  // 從特殊符號 '.' 開始

        while (true) {
            // One-hot 編碼當前字元
            Tensor xenc = Tensor.oneHot(idx, vocab.size());  // 1×27

            // 計算 logits 和機率
            Tensor logits = xenc.matmul(W);  // 1×27
            Tensor prob = TensorOps.softmax(logits);  // 1×27

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
        return "BigramNeuralNet (lr=" + learningRate + ", epochs=" + epochs + ")";
    }

    /**
     * 取得權重矩陣
     */
    public Tensor getWeights() {
        return W;
    }

    /**
     * 取得機率矩陣（將權重轉換為機率）
     */
    public Tensor getProbs() {
        Tensor probs = Tensor.zeros(vocab.size(), vocab.size());
        for (int i = 0; i < vocab.size(); i++) {
            Tensor row = W.getRow(i);
            Tensor probRow = TensorOps.softmax(row);
            for (int j = 0; j < vocab.size(); j++) {
                probs.set(i, j, probRow.get(0, j));
            }
        }
        return probs;
    }

    public List<String> sample(Random rng, int nSamples) {
        List<String> results = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            results.add(sample(rng)); // 呼叫原本的單次 sample
        }
        return results;
    }


}