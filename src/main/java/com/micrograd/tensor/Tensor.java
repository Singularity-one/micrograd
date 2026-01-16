package com.micrograd.tensor;

import java.util.Arrays;
import java.util.Random;

/**
 * 2D 張量（矩陣）
 * 用於儲存計數、機率、權重等
 */
public class Tensor {

    private final int rows;
    private final int cols;
    private final double[] data;  // row-major 儲存

    // ==================== 建構子 ====================

    public Tensor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows * cols];
    }

    public Tensor(int rows, int cols, double[] data) {
        if (data.length != rows * cols) {
            throw new IllegalArgumentException(
                    String.format("Data length %d doesn't match shape [%d, %d]",
                            data.length, rows, cols));
        }
        this.rows = rows;
        this.cols = cols;
        this.data = data.clone();
    }

    // ==================== 靜態工廠方法 ====================

    /**
     * 建立全零張量
     */
    public static Tensor zeros(int rows, int cols) {
        return new Tensor(rows, cols);
    }

    /**
     * 建立全一張量
     */
    public static Tensor ones(int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        Arrays.fill(t.data, 1.0);
        return t;
    }

    /**
     * 建立隨機張量（標準常態分佈）
     */
    public static Tensor randn(int rows, int cols, Random rng) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = rng.nextGaussian();
        }
        return t;
    }

    /**
     * 建立隨機張量（均勻分佈 [0, 1)）
     */
    public static Tensor rand(int rows, int cols, Random rng) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = rng.nextDouble();
        }
        return t;
    }

    /**
     * 從 2D 陣列建立張量
     */
    public static Tensor fromArray(double[][] arr) {
        int rows = arr.length;
        int cols = arr[0].length;
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t.set(i, j, arr[i][j]);
            }
        }
        return t;
    }

    /**
     * 建立 one-hot 向量（1×cols 的列向量）
     */
    public static Tensor oneHot(int index, int size) {
        Tensor t = new Tensor(1, size);
        t.set(0, index, 1.0);
        return t;
    }

    // ==================== 索引操作 ====================

    private int index(int row, int col) {
        return row * cols + col;
    }

    public double get(int row, int col) {
        return data[index(row, col)];
    }

    public void set(int row, int col, double value) {
        data[index(row, col)] = value;
    }

    public void increment(int row, int col, double value) {
        data[index(row, col)] += value;
    }

    /**
     * 取得某一列（回傳新的 1×cols 張量）
     */
    public Tensor getRow(int row) {
        Tensor result = new Tensor(1, cols);
        System.arraycopy(data, row * cols, result.data, 0, cols);
        return result;
    }

    /**
     * 取得某一欄（回傳新的 rows×1 張量）
     */
    public Tensor getCol(int col) {
        Tensor result = new Tensor(rows, 1);
        for (int i = 0; i < rows; i++) {
            result.set(i, 0, get(i, col));
        }
        return result;
    }

    // ==================== 形狀操作 ====================

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public int[] shape() {
        return new int[]{rows, cols};
    }

    public int size() {
        return data.length;
    }

    /**
     * 複製張量
     */
    public Tensor copy() {
        return new Tensor(rows, cols, data);
    }

    // ==================== 數學運算（回傳新張量）====================

    /**
     * 加法（元素對元素）
     */
    public Tensor add(Tensor other) {
        checkSameShape(other);
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] + other.data[i];
        }
        return result;
    }

    /**
     * 加上純量
     */
    public Tensor add(double scalar) {
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] + scalar;
        }
        return result;
    }

    /**
     * 減法（元素對元素）
     */
    public Tensor sub(Tensor other) {
        checkSameShape(other);
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] - other.data[i];
        }
        return result;
    }

    /**
     * 乘法（元素對元素）
     */
    public Tensor mul(Tensor other) {
        checkSameShape(other);
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] * other.data[i];
        }
        return result;
    }

    /**
     * 乘上純量
     */
    public Tensor mul(double scalar) {
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] * scalar;
        }
        return result;
    }

    /**
     * 除法（元素對元素）
     */
    public Tensor div(Tensor other) {
        checkSameShape(other);
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] / other.data[i];
        }
        return result;
    }

    /**
     * 除以純量
     */
    public Tensor div(double scalar) {
        return mul(1.0 / scalar);
    }

    /**
     * 矩陣乘法
     */
    public Tensor matmul(Tensor other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                    String.format("Cannot multiply [%d,%d] with [%d,%d]",
                            rows, cols, other.rows, other.cols));
        }
        Tensor result = new Tensor(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    /**
     * 轉置
     */
    public Tensor transpose() {
        Tensor result = new Tensor(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(j, i, get(i, j));
            }
        }
        return result;
    }

    // ==================== 聚合運算 ====================

    /**
     * 所有元素的總和
     */
    public double sum() {
        double sum = 0;
        for (double v : data) {
            sum += v;
        }
        return sum;
    }

    /**
     * 沿著指定軸加總
     * axis=0: 沿著列加總，結果為 1×cols
     * axis=1: 沿著欄加總，結果為 rows×1
     */
    public Tensor sum(int axis) {
        if (axis == 0) {
            Tensor result = new Tensor(1, cols);
            for (int j = 0; j < cols; j++) {
                double sum = 0;
                for (int i = 0; i < rows; i++) {
                    sum += get(i, j);
                }
                result.set(0, j, sum);
            }
            return result;
        } else if (axis == 1) {
            Tensor result = new Tensor(rows, 1);
            for (int i = 0; i < rows; i++) {
                double sum = 0;
                for (int j = 0; j < cols; j++) {
                    sum += get(i, j);
                }
                result.set(i, 0, sum);
            }
            return result;
        } else {
            throw new IllegalArgumentException("Axis must be 0 or 1");
        }
    }

    /**
     * 所有元素的平均
     */
    public double mean() {
        return sum() / data.length;
    }

    /**
     * 找最大值的索引（用於 1D 或單列張量）
     */
    public int argmax() {
        int maxIdx = 0;
        double maxVal = data[0];
        for (int i = 1; i < data.length; i++) {
            if (data[i] > maxVal) {
                maxVal = data[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    // ==================== 元素運算 ====================

    /**
     * 指數（e^x）
     */
    public Tensor exp() {
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = Math.exp(this.data[i]);
        }
        return result;
    }

    /**
     * 自然對數
     */
    public Tensor log() {
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = Math.log(this.data[i]);
        }
        return result;
    }

    /**
     * 負號
     */
    public Tensor neg() {
        return mul(-1);
    }

    // ==================== 廣播除法 ====================

    /**
     * 每列除以對應的值（用於正規化）
     * divisor 必須是 rows×1 的張量
     */
    public Tensor divByColumn(Tensor divisor) {
        if (divisor.rows != this.rows || divisor.cols != 1) {
            throw new IllegalArgumentException(
                    String.format("Divisor shape [%d,%d] incompatible with [%d,%d]",
                            divisor.rows, divisor.cols, rows, cols));
        }
        Tensor result = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++) {
            double d = divisor.get(i, 0);
            for (int j = 0; j < cols; j++) {
                result.set(i, j, get(i, j) / d);
            }
        }
        return result;
    }

    // ==================== 輔助方法 ====================

    private void checkSameShape(Tensor other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException(
                    String.format("Shape mismatch: [%d,%d] vs [%d,%d]",
                            rows, cols, other.rows, other.cols));
        }
    }

    /**
     * 轉換為原始陣列
     */
    public double[] toArray() {
        return data.clone();
    }

    /**
     * 轉換為 2D 陣列
     */
    public double[][] to2DArray() {
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = get(i, j);
            }
        }
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Tensor [%d, %d]:\n", rows, cols));
        for (int i = 0; i < rows; i++) {
            sb.append("[");
            for (int j = 0; j < cols; j++) {
                sb.append(String.format("%8.4f", get(i, j)));
                if (j < cols - 1) sb.append(", ");
            }
            sb.append("]\n");
        }
        return sb.toString();
    }

    /**
     * 簡短的字串表示（只顯示形狀和部分資料）
     */
    public String toShortString() {
        return String.format("Tensor [%d, %d], sum=%.4f", rows, cols, sum());
    }
}