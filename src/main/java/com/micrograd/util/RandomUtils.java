package com.micrograd.util;

import com.micrograd.tensor.Tensor;
import java.util.Random;

/**
 * 隨機採樣工具
 */
public class RandomUtils {

    private RandomUtils() {
        // 工具類別，不允許實例化
    }

    /**
     * 根據機率分佈採樣一個索引
     * 實現 torch.multinomial 的功能
     *
     * @param probs 機率分佈（必須加總為 1）
     * @param rng   隨機數生成器
     * @return 採樣得到的索引
     */
    public static int multinomial(Tensor probs, Random rng) {
        double r = rng.nextDouble();
        double cumsum = 0;

        int size = probs.getCols();
        for (int i = 0; i < size; i++) {
            cumsum += probs.get(0, i);
            if (r < cumsum) {
                return i;
            }
        }

        // 由於浮點數精度問題，可能會到達這裡
        return size - 1;
    }

    /**
     * 根據機率分佈採樣一個索引（從 double 陣列）
     */
    public static int multinomial(double[] probs, Random rng) {
        double r = rng.nextDouble();
        double cumsum = 0;

        for (int i = 0; i < probs.length; i++) {
            cumsum += probs[i];
            if (r < cumsum) {
                return i;
            }
        }

        return probs.length - 1;
    }

    /**
     * 驗證機率分佈是否有效（加總約為 1）
     */
    public static boolean isValidProbDistribution(Tensor probs, double tolerance) {
        double sum = probs.sum();
        return Math.abs(sum - 1.0) < tolerance;
    }

    /**
     * 驗證機率分佈是否有效（預設容忍度）
     */
    public static boolean isValidProbDistribution(Tensor probs) {
        return isValidProbDistribution(probs, 1e-6);
    }
}