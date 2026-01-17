package com.micrograd;

import java.io.IOException;
import java.util.*;
import com.micrograd.model.*;
import com.micrograd.data.*;
import com.micrograd.tensor.*;
import com.micrograd.util.BigramVisualizer;

public class Main {

    public static void main(String[] args) {

        // === 範例 1：簡單表達式 ===
//        Value a = new Value(2.0);
//        a.setLabel("a");
//
//        Value b = new Value(-3.0);
//        b.setLabel("b");
//
//        Value c = new Value(10.0);
//        c.setLabel("c");
//
//        Value e = a.mul(b);
//        e.setLabel("e");
//
//        Value d = e.add(c);
//        d.setLabel("d");
//
//        Value f = new Value(-2.0);
//        f.setLabel("f");
//
//        Value L = d.mul(f);
//        L.setLabel("L");
//
//        // 反向傳播
//        L.backward();
//
//        // 視覺化（自動開啟圖片）
//        GraphVisualizer.draw(L, "example1_simple");
//
//
//        // === 範例 2：神經元 ===
//        Value x1 = new Value(2.0);
//        x1.setLabel("x1");
//
//        Value x2 = new Value(0.0);
//        x2.setLabel("x2");
//
//        Value w1 = new Value(-3.0);
//        w1.setLabel("w1");
//
//        Value w2 = new Value(1.0);
//        w2.setLabel("w2");
//
//        Value bias = new Value(6.8813735870195432);
//        bias.setLabel("b");
//
//        Value x1w1 = x1.mul(w1);
//        x1w1.setLabel("x1*w1");
//
//        Value x2w2 = x2.mul(w2);
//        x2w2.setLabel("x2*w2");
//
//        Value x1w1x2w2 = x1w1.add(x2w2);
//        x1w1x2w2.setLabel("x1w1+x2w2");
//
//        Value n = x1w1x2w2.add(bias);
//        n.setLabel("n");
//
//        Value o = n.tanh();
//        o.setLabel("o");
//
//        // 反向傳播
//        o.backward();
//
//        // 視覺化
//        GraphVisualizer.draw(o, "example2_neuron");
//
//        System.out.println("\n=== 神經元梯度 ===");
//        System.out.println("x1.grad = " + x1.getGrad());
//        System.out.println("x2.grad = " + x2.getGrad());
//        System.out.println("w1.grad = " + w1.getGrad());
//        System.out.println("w2.grad = " + w2.getGrad());

        // === 範例 2 ===
        try {
            runCompleteDemonstration();
        } catch (IOException e) {
            System.err.println("錯誤: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void runCompleteDemonstration() throws IOException {

        BigramVisualizer.printHeader("Java Makemore Part 1: Bigram Language Model");

        // ========================================
        // Part 1: 資料載入和探索
        // ========================================

        BigramVisualizer.printSubHeader("Part 1: Loading and Exploring Data");

        // 建立詞彙表
        Vocabulary vocab = new Vocabulary();
        System.out.println("Vocabulary size: " + vocab.size());
        System.out.println("Special token: '" + Vocabulary.SPECIAL_TOKEN + "' at index 0");
        System.out.println();

        // 載入資料
        String dataPath = "data/names.txt";
        DataLoader loader = new DataLoader(dataPath, vocab);
        loader.printStats();

        // ========================================
        // Part 2: Bigram 計數過程展示
        // ========================================

        BigramVisualizer.printHeader("Part 2: Building Bigrams");

        // 展示單個單字的 bigram
        BigramVisualizer.printBigramBuildingProcess("emma", vocab);
        BigramVisualizer.printBigramBuildingProcess("olivia", vocab);

        // 展示多個單字
        BigramVisualizer.printMultipleBigrams(
                loader.getWords().subList(0, 5), vocab, 5);

        // ========================================
        // Part 3: 統計方法 - Counting
        // ========================================

        BigramVisualizer.printHeader("Part 3: Counting Method");

        BigramCounter counterModel = new BigramCounter(vocab, 1.0);
        int[][] bigrams = loader.getBigrams();

        System.out.println("Training counting model...");
        counterModel.train(bigrams);
        System.out.println("✓ Training complete\n");

        // 顯示 counts 矩陣
        BigramVisualizer.printCountsMatrixSample(counterModel.getCounts(), vocab, 8);

        // 顯示最常見的 bigrams
        BigramVisualizer.printTopBigrams(counterModel.getCounts(), vocab, 15);

        // ========================================
        // Part 4: 機率和正規化
        // ========================================

        BigramVisualizer.printHeader("Part 4: Probabilities and Normalization");

        boolean normalized = counterModel.checkNormalization();
        System.out.println("Normalization check: " + (normalized ? "✓ PASS" : "✗ FAIL"));
        System.out.println();

        // 顯示某些字元之後最可能的字元
        counterModel.printTopNext('.', 5);
        System.out.println();
        counterModel.printTopNext('a', 5);
        System.out.println();
        counterModel.printTopNext('e', 5);

        // ========================================
        // Part 5: Loss 計算詳解
        // ========================================

        BigramVisualizer.printHeader("Part 5: Loss Calculation");

        // 逐個 bigram 計算 loss
        BigramVisualizer.printDetailedLossCalculation(
                "emma", vocab, counterModel.getProbs());

        BigramVisualizer.printDetailedLossCalculation(
                "olivia", vocab, counterModel.getProbs());

        // 整體 loss
        double counterLoss = counterModel.loss(bigrams);
        System.out.println();
        BigramVisualizer.printSubHeader("Overall Training Loss");
        System.out.printf("Total training examples: %d\n", bigrams[0].length);
        System.out.printf("Average Negative Log-Likelihood: %.4f\n", counterLoss);

        // Loss 分佈
        BigramVisualizer.printLossDistribution(bigrams, counterModel.getProbs(), 10);

        // ========================================
        // Part 6: Sampling 過程展示
        // ========================================

        BigramVisualizer.printHeader("Part 6: Sampling from the Model");

        // 逐步展示 sampling
        Random rng1 = new Random(42);
        BigramVisualizer.sampleWithVisualization(
                counterModel.getProbs(), vocab, rng1);

        // 生成多個名字
        BigramVisualizer.printSubHeader("Generated Names (Counting Method)");
        Random rng2 = new Random(42);
        List<String> counterSamples = counterModel.sampleMultiple(rng2, 20);

        System.out.println("Generated 20 names:");
        for (int i = 0; i < counterSamples.size(); i++) {
            System.out.printf("%2d. %s\n", i + 1, counterSamples.get(i));
        }

        // ========================================
        // Part 7: 神經網路方法
        // ========================================

        BigramVisualizer.printHeader("Part 7: Neural Network Approach");

        // One-hot encoding 展示
        BigramVisualizer.printOneHotExample(vocab);

        // 訓練神經網路
        BigramVisualizer.printSubHeader("Training Neural Network");

        Random initRng = new Random(42);
        BigramNeuralNet nnModel = new BigramNeuralNet(vocab, 50.0, 100, initRng);

        System.out.println("Configuration:");
        System.out.println("  Learning rate: 50.0");
        System.out.println("  Epochs: 100");
        System.out.println("  Weight initialization: Random (seed=42)");
        System.out.println();

        // 訓練（內部已有進度顯示）
        nnModel.train(bigrams);

        double nnLoss = nnModel.loss(bigrams);
        System.out.println();
        System.out.printf("Final training loss: %.4f\n", nnLoss);

        // ========================================
        // Part 8: 生成名字（神經網路）
        // ========================================

        BigramVisualizer.printSubHeader("Generated Names (Neural Network)");

        Random rng3 = new Random(42);
        List<String> nnSamples = nnModel.sample(rng3, 20);

        System.out.println("Generated 20 names:");
        for (int i = 0; i < nnSamples.size(); i++) {
            System.out.printf("%2d. %s\n", i + 1, nnSamples.get(i));
        }

        // ========================================
        // Part 9: 兩種方法的比較
        // ========================================

        BigramVisualizer.printHeader("Part 9: Comparing Both Approaches");

        // Loss 比較
        BigramVisualizer.compareLoss(counterLoss, nnLoss);

        // 機率矩陣比較
        BigramVisualizer.compareModels(counterModel, nnModel, vocab);

        // ========================================
        // Part 10: 並排比較生成結果
        // ========================================

        BigramVisualizer.printSubHeader("Side-by-Side Comparison");

        System.out.println("Same random seed (42) produces same results:");
        System.out.println();
        System.out.printf("%-5s %-20s %-20s\n", "No.", "Counting", "Neural Net");
        System.out.println("-".repeat(50));

        for (int i = 0; i < Math.min(15, counterSamples.size()); i++) {
            System.out.printf("%-5d %-20s %-20s %s\n",
                    i + 1,
                    counterSamples.get(i),
                    nnSamples.get(i),
                    counterSamples.get(i).equals(nnSamples.get(i)) ? "✓" : "");
        }

        // ========================================
        // 結論
        // ========================================

        BigramVisualizer.printHeader("Conclusion");

        System.out.println("Key Takeaways:");
        System.out.println();
        System.out.println("1. 兩種方法本質上是等價的");
        System.out.println("   - 計數方法：直接統計並正規化");
        System.out.println("   - 神經網路：通過梯度下降學習相同的機率分佈");
        System.out.println();
        System.out.println("2. 神經網路方法的優勢：");
        System.out.println("   - 可以擴展到更複雜的模型（MLP, RNN, Transformer）");
        System.out.println("   - 可以處理更長的 context");
        System.out.println("   - 不需要儲存龐大的計數表");
        System.out.println();
        System.out.println("3. Negative Log-Likelihood 作為損失函數：");
        System.out.printf("   - Counting:  %.4f\n", counterLoss);
        System.out.printf("   - Neural:    %.4f\n", nnLoss);
        System.out.printf("   - Difference: %.4f (%.2f%%)\n",
                Math.abs(counterLoss - nnLoss),
                100.0 * Math.abs(counterLoss - nnLoss) / counterLoss);
        System.out.println();
        System.out.println("4. 下一步：");
        System.out.println("   - 使用更長的 context (trigrams, n-grams)");
        System.out.println("   - MLP-based language model");
        System.out.println("   - 探索不同的網路架構");
        System.out.println();

        BigramVisualizer.printHeader("makemore Part 1 Complete!");
    }
}