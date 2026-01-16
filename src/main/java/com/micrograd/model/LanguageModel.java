package com.micrograd.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 語言模型介面
 */
public interface LanguageModel {

    void train(int[][] bigrams);

    double loss(int[][] bigrams);

    String sample(Random rng);

    /**
     * 生成多個新名字
     */
    default List<String> sampleMultiple(Random rng, int count) {
        List<String> result = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            result.add(sample(rng));
        }
        return result;
    }

    String getName();
}