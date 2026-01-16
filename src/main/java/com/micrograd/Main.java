package com.micrograd;

import java.io.IOException;
import java.util.*;
import com.micrograd.model.*;
import com.micrograd.data.*;
import com.micrograd.tensor.*;

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
            run();
        } catch (IOException e) {
            System.err.println("錯誤: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void run() throws IOException {
        System.out.println("========================================");
        System.out.println("   Java Makemore - Bigram 語言模型");
        System.out.println("========================================\n");

        // 1. 建立詞彙表
        Vocabulary vocab = new Vocabulary();
        System.out.println(vocab);

        // 2. 載入資料
        String dataPath = "data/names.txt";
        DataLoader loader = new DataLoader(dataPath, vocab);
        loader.printStats();
        System.out.println();

        // 印出範例 bigram
        loader.printBigrams("emma");
        System.out.println();

        // 3. 取得 bigram 資料
        int[][] bigrams = loader.getBigrams();
        System.out.println("總 Bigram 數量: " + bigrams[0].length);
        System.out.println();

        // 4. 方法一：計數統計
        System.out.println("========================================");
        System.out.println("   方法一：計數統計");
        System.out.println("========================================\n");

        BigramCounter counterModel = new BigramCounter(vocab, 1.0);
        counterModel.train(bigrams);

        System.out.println("機率分佈正規化檢查: " +
                (counterModel.checkNormalization() ? "通過" : "失敗"));
        System.out.println();

        double counterLoss = counterModel.loss(bigrams);
        System.out.printf("訓練集損失: %.4f\n\n", counterLoss);

        // 印出某些字元之後最可能的字元
        counterModel.printTopNext('.', 5);
        System.out.println();
        counterModel.printTopNext('a', 5);
        System.out.println();

        // 生成名字
        System.out.println("生成的名字（計數方法）:");
        Random rng1 = new Random(42);
        List<String> counterSamples = counterModel.sampleMultiple(rng1, 10);
        for (String name : counterSamples) {
            System.out.println("  " + name);
        }


        // 5. 方法二：神經網路
        System.out.println("========================================");
        System.out.println("   方法二：神經網路");
        System.out.println("========================================\n");

        Random initRng = new Random(42);
        BigramNeuralNet nnModel = new BigramNeuralNet(vocab, 50.0, 100, initRng);
        nnModel.train(bigrams);
        System.out.println();

        double nnLoss = nnModel.loss(bigrams);
        System.out.printf("訓練集損失: %.4f\n\n", nnLoss);

        // 生成名字
        System.out.println("生成的名字（神經網路方法）:");
        Random rng2 = new Random(42);
        List<String> nnSamples = nnModel.sample(rng2, 10);
        for (String name : nnSamples) {
            System.out.println("  " + name);
        }
        System.out.println();

        // 6. 比較兩種方法
        System.out.println("========================================");
        System.out.println("   結果比較");
        System.out.println("========================================\n");

        System.out.printf("計數方法損失: %.4f\n", counterLoss);
        System.out.printf("神經網路損失: %.4f\n", nnLoss);
        System.out.printf("差異: %.4f\n", Math.abs(counterLoss - nnLoss));
        System.out.println();

        System.out.println("結論: 兩種方法學到的機率分佈應該非常接近！");
    }
}