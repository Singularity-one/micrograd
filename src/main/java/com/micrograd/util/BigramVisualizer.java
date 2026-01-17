package com.micrograd.util;

import com.micrograd.data.Vocabulary;
import com.micrograd.model.BigramCounter;
import com.micrograd.model.BigramNeuralNet;
import com.micrograd.tensor.Tensor;

import java.util.*;

/**
 * Bigram 語言模型視覺化工具
 * 提供教學展示功能，對應 Karpathy makemore Part 1 的所有視覺化
 */
public class BigramVisualizer {

    private static final int CONSOLE_WIDTH = 80;
    private static final String SEPARATOR = "=".repeat(CONSOLE_WIDTH);
    private static final String THIN_SEPARATOR = "-".repeat(CONSOLE_WIDTH);

    // ==================== 標題列印 ====================

    public static void printHeader(String title) {
        System.out.println("\n" + SEPARATOR);
        System.out.println(centerText(title));
        System.out.println(SEPARATOR + "\n");
    }

    public static void printSubHeader(String title) {
        System.out.println("\n" + THIN_SEPARATOR);
        System.out.println(title);
        System.out.println(THIN_SEPARATOR);
    }

    private static String centerText(String text) {
        int padding = (CONSOLE_WIDTH - text.length()) / 2;
        return " ".repeat(Math.max(0, padding)) + text;
    }

    // ==================== 1. Bigram 計數過程展示 ====================

    /**
     * 逐步展示單字的 bigram 計數過程
     */
    public static void printBigramBuildingProcess(String word, Vocabulary vocab) {
        printSubHeader("Building Bigrams for: \"" + word + "\"");

        // 加上開始和結束符號
        String padded = Vocabulary.SPECIAL_TOKEN + word + Vocabulary.SPECIAL_TOKEN;

        System.out.println("原始單字: " + word);
        System.out.println("加上特殊符號: " + displayString(padded));
        System.out.println("\nBigrams:");

        for (int i = 0; i < padded.length() - 1; i++) {
            char ch1 = padded.charAt(i);
            char ch2 = padded.charAt(i + 1);
            int idx1 = vocab.encode(ch1);
            int idx2 = vocab.encode(ch2);

            String display1 = (ch1 == '.') ? "START" : String.valueOf(ch1);
            String display2 = (ch2 == '.') ? "END" : String.valueOf(ch2);

            System.out.printf("  %5s (%2d) → %5s (%2d)\n",
                    display1, idx1, display2, idx2);
        }
    }

    /**
     * 展示多個單字的 bigram
     */
    public static void printMultipleBigrams(List<String> words, Vocabulary vocab, int maxWords) {
        printSubHeader("Bigram Examples from Training Data");

        int count = 0;
        for (String word : words) {
            if (count >= maxWords) break;
            System.out.println("\n\"" + word + "\":");
            String padded = Vocabulary.SPECIAL_TOKEN + word + Vocabulary.SPECIAL_TOKEN;

            for (int i = 0; i < padded.length() - 1; i++) {
                char ch1 = padded.charAt(i);
                char ch2 = padded.charAt(i + 1);
                String display1 = (ch1 == '.') ? "." : String.valueOf(ch1);
                String display2 = (ch2 == '.') ? "." : String.valueOf(ch2);
                System.out.print("  " + display1 + display2);
            }
            System.out.println();
            count++;
        }
    }

    // ==================== 2. Counts 矩陣視覺化 ====================

    /**
     * 顯示 counts 矩陣的部分內容（ASCII art 風格）
     */
    public static void printCountsMatrixSample(Tensor counts, Vocabulary vocab, int size) {
        printSubHeader("Counts Matrix Sample (Top-left " + size + "x" + size + ")");

        // 印出列標題
        System.out.print("     ");
        for (int j = 0; j < size; j++) {
            char ch = vocab.decode(j);
            String display = (ch == '.') ? "." : String.valueOf(ch);
            System.out.printf(" %4s", display);
        }
        System.out.println();
        System.out.println("    " + "-".repeat(size * 5 + 1));

        // 印出矩陣內容
        for (int i = 0; i < size; i++) {
            char ch = vocab.decode(i);
            String display = (ch == '.') ? "." : String.valueOf(ch);
            System.out.printf(" %2s |", display);

            for (int j = 0; j < size; j++) {
                int count = (int) counts.get(i, j);
                if (count == 0) {
                    System.out.print("    .");
                } else {
                    System.out.printf(" %4d", count);
                }
            }
            System.out.println();
        }
    }

    /**
     * 顯示最常見的 bigrams
     */
    public static void printTopBigrams(Tensor counts, Vocabulary vocab, int topK) {
        printSubHeader("Top " + topK + " Most Frequent Bigrams");

        // 收集所有 non-zero counts
        List<BigramCount> bigramCounts = new ArrayList<>();
        for (int i = 0; i < vocab.size(); i++) {
            for (int j = 0; j < vocab.size(); j++) {
                double count = counts.get(i, j);
                if (count > 0) {
                    bigramCounts.add(new BigramCount(i, j, (int) count));
                }
            }
        }

        // 排序
        bigramCounts.sort((a, b) -> Integer.compare(b.count, a.count));

        // 顯示前 K 個
        System.out.printf("%-10s %10s %15s\n", "Bigram", "Count", "Percentage");
        System.out.println(THIN_SEPARATOR);

        int totalCount = (int) counts.sum();
        for (int i = 0; i < Math.min(topK, bigramCounts.size()); i++) {
            BigramCount bc = bigramCounts.get(i);
            char ch1 = vocab.decode(bc.idx1);
            char ch2 = vocab.decode(bc.idx2);

            String display1 = (ch1 == '.') ? "." : String.valueOf(ch1);
            String display2 = (ch2 == '.') ? "." : String.valueOf(ch2);
            String bigram = display1 + display2;

            double percentage = 100.0 * bc.count / totalCount;
            System.out.printf("%-10s %10d %14.2f%%\n", bigram, bc.count, percentage);
        }
    }

    // ==================== 3. Loss 計算詳細展示 ====================

    /**
     * 逐個 bigram 展示 loss 計算過程
     */
    public static void printDetailedLossCalculation(
            String word, Vocabulary vocab, Tensor probs) {

        printSubHeader("Detailed Loss Calculation for: \"" + word + "\"");

        String padded = Vocabulary.SPECIAL_TOKEN + word + Vocabulary.SPECIAL_TOKEN;
        double totalLogProb = 0.0;
        int n = 0;

        System.out.printf("%-8s %12s %12s %12s\n",
                "Bigram", "Prob", "Log(Prob)", "NLL");
        System.out.println(THIN_SEPARATOR);

        for (int i = 0; i < padded.length() - 1; i++) {
            char ch1 = padded.charAt(i);
            char ch2 = padded.charAt(i + 1);
            int idx1 = vocab.encode(ch1);
            int idx2 = vocab.encode(ch2);

            double prob = probs.get(idx1, idx2);
            double logProb = Math.log(prob);
            double nll = -logProb;

            String display1 = (ch1 == '.') ? "." : String.valueOf(ch1);
            String display2 = (ch2 == '.') ? "." : String.valueOf(ch2);
            String bigram = display1 + display2;

            System.out.printf("%-8s %12.6f %12.4f %12.4f\n",
                    bigram, prob, logProb, nll);

            totalLogProb += logProb;
            n++;
        }

        double avgNLL = -totalLogProb / n;
        System.out.println(THIN_SEPARATOR);
        System.out.printf("Average Negative Log-Likelihood: %.4f\n", avgNLL);
    }

    /**
     * 展示整個資料集的 loss 分佈
     */
    public static void printLossDistribution(int[][] bigrams, Tensor probs, int samples) {
        printSubHeader("Loss Distribution (First " + samples + " Bigrams)");

        System.out.printf("%-6s %12s %12s\n", "Index", "Prob", "NLL");
        System.out.println(THIN_SEPARATOR);

        double totalLoss = 0.0;
        int count = Math.min(samples, bigrams[0].length);

        for (int i = 0; i < count; i++) {
            int idx1 = bigrams[0][i];
            int idx2 = bigrams[1][i];

            double prob = probs.get(idx1, idx2);
            double nll = -Math.log(prob);

            System.out.printf("%-6d %12.6f %12.4f\n", i, prob, nll);
            totalLoss += nll;
        }

        System.out.println(THIN_SEPARATOR);
        System.out.printf("Average NLL (first %d): %.4f\n", count, totalLoss / count);
    }

    // ==================== 4. Sampling 過程展示 ====================

    /**
     * 逐步展示 sampling 過程
     */
    public static String sampleWithVisualization(Tensor probs, Vocabulary vocab, Random rng) {
        printSubHeader("Step-by-Step Sampling Process");

        StringBuilder result = new StringBuilder();
        int idx = 0;  // 從 START token 開始
        int step = 0;

        System.out.printf("%-6s %-10s %-15s %s\n",
                "Step", "Current", "Sampled", "Top-3 Probs");
        System.out.println(THIN_SEPARATOR);

        while (true) {
            // 取得當前的機率分佈
            Tensor prob = probs.getRow(idx);

            // 顯示 top-3 機率
            String top3 = getTop3Probs(prob, vocab);

            // 採樣
            idx = RandomUtils.multinomial(prob, rng);

            char currentChar = vocab.decode(idx);
            String display = (currentChar == '.') ? "END" : String.valueOf(currentChar);

            System.out.printf("%-6d %-10s %-15s %s\n",
                    step,
                    (step == 0 ? "START" : String.valueOf(result.charAt(result.length() - 1))),
                    display,
                    top3);

            if (idx == 0) break;  // END token

            result.append(currentChar);
            step++;

            if (step > 20) {  // 安全限制
                System.out.println("(Stopped: max length reached)");
                break;
            }
        }

        System.out.println(THIN_SEPARATOR);
        System.out.println("Generated name: \"" + result + "\"");
        return result.toString();
    }

    /**
     * 取得機率分佈的 top-3
     */
    private static String getTop3Probs(Tensor prob, Vocabulary vocab) {
        List<ProbPair> pairs = new ArrayList<>();
        for (int i = 0; i < prob.getCols(); i++) {
            pairs.add(new ProbPair(i, prob.get(0, i)));
        }
        pairs.sort((a, b) -> Double.compare(b.prob, a.prob));

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Math.min(3, pairs.size()); i++) {
            if (i > 0) sb.append(", ");
            char ch = vocab.decode(pairs.get(i).idx);
            String display = (ch == '.') ? "." : String.valueOf(ch);
            sb.append(String.format("%s:%.2f", display, pairs.get(i).prob));
        }
        return sb.toString();
    }

    // ==================== 5. One-hot Encoding 展示 ====================

    /**
     * 顯示 one-hot encoding 範例
     */
    public static void printOneHotExample(Vocabulary vocab) {
        printSubHeader("One-Hot Encoding Examples");

        int[] examples = {0, 5, 13};  // '.', 'e', 'm'
        for (int idx : examples) {
            char ch = vocab.decode(idx);
            String display = (ch == '.') ? "START/END" : String.valueOf(ch);

            System.out.println("\nCharacter: '" + display + "' (index=" + idx + ")");
            System.out.print("One-hot:   [");

            for (int i = 0; i < vocab.size(); i++) {
                if (i == idx) {
                    System.out.print("1");
                } else {
                    System.out.print("0");
                }
                if (i < vocab.size() - 1) {
                    System.out.print(i % 5 == 4 ? " | " : " ");
                }
            }
            System.out.println("]");
        }
    }

    // ==================== 6. 模型比較 ====================

    /**
     * 比較兩個模型的機率矩陣
     */
    public static void compareModels(BigramCounter counter, BigramNeuralNet nn, Vocabulary vocab) {
        printSubHeader("Comparing Counting vs Neural Network Models");

        Tensor probsCounter = counter.getProbs();
        Tensor probsNN = nn.getProbs();

        // 顯示部分機率值
        System.out.println("Sample probabilities (first 5 columns):");
        System.out.println("\nRow 0 (after START token):");
        System.out.print("Counter:  ");
        for (int j = 0; j < 5; j++) {
            System.out.printf("%.4f ", probsCounter.get(0, j));
        }
        System.out.print("\nNeural:   ");
        for (int j = 0; j < 5; j++) {
            System.out.printf("%.4f ", probsNN.get(0, j));
        }
        System.out.println("\n");

        // 計算最大差異
        double maxDiff = 0.0;
        double sumDiff = 0.0;
        int count = 0;

        for (int i = 0; i < vocab.size(); i++) {
            for (int j = 0; j < vocab.size(); j++) {
                double diff = Math.abs(probsCounter.get(i, j) - probsNN.get(i, j));
                maxDiff = Math.max(maxDiff, diff);
                sumDiff += diff;
                count++;
            }
        }

        double avgDiff = sumDiff / count;

        System.out.println("Probability Matrix Comparison:");
        System.out.printf("  Maximum difference: %.6f\n", maxDiff);
        System.out.printf("  Average difference: %.6f\n", avgDiff);
        System.out.printf("  Are they equivalent? %s\n",
                maxDiff < 0.01 ? "✓ YES" : "✗ NO");
    }

    /**
     * 比較損失值
     */
    public static void compareLoss(double lossCounter, double lossNN) {
        printSubHeader("Loss Comparison");

        System.out.printf("Counting method loss:      %.6f\n", lossCounter);
        System.out.printf("Neural network loss:       %.6f\n", lossNN);
        System.out.printf("Difference:                %.6f\n", Math.abs(lossCounter - lossNN));
        System.out.printf("Relative difference:       %.4f%%\n",
                100.0 * Math.abs(lossCounter - lossNN) / lossCounter);
    }

    // ==================== 7. 訓練過程視覺化 ====================

    /**
     * 顯示訓練進度（在訓練循環中調用）
     */
    public static void printTrainingProgress(int epoch, int totalEpochs, double loss) {
        if (epoch == 0 || epoch == totalEpochs - 1 || epoch % 10 == 0) {
            System.out.printf("Epoch %4d/%d | Loss: %.6f\n", epoch, totalEpochs, loss);
        }
    }

    /**
     * 訓練總結
     */
    public static void printTrainingSummary(int epochs, double initialLoss, double finalLoss) {
        printSubHeader("Training Summary");

        System.out.printf("Total epochs:        %d\n", epochs);
        System.out.printf("Initial loss:        %.6f\n", initialLoss);
        System.out.printf("Final loss:          %.6f\n", finalLoss);
        System.out.printf("Loss reduction:      %.6f (%.2f%%)\n",
                initialLoss - finalLoss,
                100.0 * (initialLoss - finalLoss) / initialLoss);
    }

    // ==================== 輔助資料結構 ====================

    private static class BigramCount {
        int idx1, idx2, count;

        BigramCount(int idx1, int idx2, int count) {
            this.idx1 = idx1;
            this.idx2 = idx2;
            this.count = count;
        }
    }

    private static class ProbPair {
        int idx;
        double prob;

        ProbPair(int idx, double prob) {
            this.idx = idx;
            this.prob = prob;
        }
    }

    // ==================== 輔助方法 ====================

    private static String displayString(String s) {
        return s.replace('.', '•');  // 用 • 代替 . 更清楚
    }
}