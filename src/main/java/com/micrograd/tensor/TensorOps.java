package com.micrograd.tensor;

/**
 * 張量運算工具類別
 */
public class TensorOps {

    private TensorOps() {
        // 工具類別，不允許實例化
    }

    /**
     * Softmax：將 logits 轉換為機率分佈
     * 對每一列獨立做 softmax
     */
    public static Tensor softmax(Tensor logits) {
        int rows = logits.getRows();
        int cols = logits.getCols();
        Tensor result = new Tensor(rows, cols);

        for (int i = 0; i < rows; i++) {
            // 為了數值穩定性，先減去最大值
            double maxVal = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) {
                maxVal = Math.max(maxVal, logits.get(i, j));
            }

            // 計算 exp(x - max)
            double sumExp = 0;
            for (int j = 0; j < cols; j++) {
                double expVal = Math.exp(logits.get(i, j) - maxVal);
                result.set(i, j, expVal);
                sumExp += expVal;
            }

            // 正規化
            for (int j = 0; j < cols; j++) {
                result.set(i, j, result.get(i, j) / sumExp);
            }
        }

        return result;
    }

    /**
     * 將計數矩陣正規化為機率分佈（每列加總為 1）
     */
    public static Tensor normalizeRows(Tensor counts) {
        Tensor out = Tensor.zeros(counts.getRows(), counts.getCols());

        for (int i = 0; i < counts.getRows(); i++) {
            double rowSum = counts.getRow(i).sum();

            if (rowSum == 0.0) {
                // smoothing = 0，且該 row 從未出現
                // 保持全 0（合法機率分佈的「退化情況」）
                continue;
            }

            for (int j = 0; j < counts.getCols(); j++) {
                out.set(i, j, counts.get(i, j) / rowSum);
            }
        }
        return out;
    }

    /**
     * 建立 one-hot 編碼矩陣
     * 給定索引陣列，建立 n×vocabSize 的 one-hot 矩陣
     */
    public static Tensor oneHotBatch(int[] indices, int vocabSize) {
        int n = indices.length;
        Tensor result = Tensor.zeros(n, vocabSize);
        for (int i = 0; i < n; i++) {
            result.set(i, indices[i], 1.0);
        }
        return result;
    }

    /**
     * 計算負對數似然損失
     * probs: 機率矩陣 (n × vocabSize)
     * targets: 目標索引陣列 (長度 n)
     * 回傳平均損失
     */
    public static double negativeLogLikelihood(Tensor probs, int[] targets) {
        double totalLoss = 0;
        int n = targets.length;

        for (int i = 0; i < n; i++) {
            int targetIdx = targets[i];
            double prob = probs.get(i, targetIdx);
            totalLoss += -Math.log(prob);
        }

        return totalLoss / n;
    }

    /**
     * 計算單一樣本的負對數似然
     */
    public static double negativeLogLikelihood(Tensor prob, int target) {
        return -Math.log(prob.get(0, target));
    }

    /**
     * 從機率分佈中取得指定索引的機率
     * probs: 1×vocabSize 的機率向量
     * index: 目標索引
     */
    public static double getProb(Tensor probs, int index) {
        return probs.get(0, index);
    }
}