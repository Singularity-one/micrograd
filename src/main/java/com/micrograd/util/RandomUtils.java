package com.micrograd.util;

import com.micrograd.tensor.Tensor;
import java.util.Random;

/**
 * 隨機數工具類別
 * 提供從機率分佈採樣的功能
 */
public class RandomUtils {

    /**
     * 從離散機率分佈中採樣
     *
     * @param probs 機率向量（應該是 1×n 或 n×1 的 Tensor，總和為 1）
     * @param rng 隨機數生成器
     * @return 採樣得到的索引
     */
    public static int multinomial(Tensor probs, Random rng) {
        // 確保是 1D 向量
        if (probs.getRows() != 1 && probs.getCols() != 1) {
            throw new IllegalArgumentException(
                    "Probability tensor must be 1D, got shape: [" +
                            probs.getRows() + ", " + probs.getCols() + "]");
        }

        // 取得機率數組
        int n = Math.max(probs.getRows(), probs.getCols());
        double[] probArray = new double[n];

        if (probs.getRows() == 1) {
            // 行向量 1×n
            for (int i = 0; i < n; i++) {
                probArray[i] = probs.get(0, i);
            }
        } else {
            // 列向量 n×1
            for (int i = 0; i < n; i++) {
                probArray[i] = probs.get(i, 0);
            }
        }

        // 檢查機率是否正規化（允許一些浮點誤差）
        double sum = 0;
        for (double p : probArray) {
            sum += p;
        }
        if (Math.abs(sum - 1.0) > 1e-5) {
            throw new IllegalArgumentException(
                    "Probabilities must sum to 1.0, got: " + sum);
        }

        // 採樣
        double r = rng.nextDouble();
        double cumulativeProb = 0.0;

        for (int i = 0; i < n; i++) {
            cumulativeProb += probArray[i];
            if (r < cumulativeProb) {
                return i;
            }
        }

        // 由於浮點誤差，可能到這裡，返回最後一個
        return n - 1;
    }

    /**
     * 從標準常態分佈採樣
     */
    public static double randn(Random rng) {
        return rng.nextGaussian();
    }

    /**
     * 從均勻分佈 [0, 1) 採樣
     */
    public static double rand(Random rng) {
        return rng.nextDouble();
    }

    /**
     * 從均勻分佈 [min, max) 採樣
     */
    public static double uniform(Random rng, double min, double max) {
        return min + (max - min) * rng.nextDouble();
    }
}