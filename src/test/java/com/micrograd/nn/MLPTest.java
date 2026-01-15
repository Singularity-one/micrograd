package com.micrograd.nn;

import com.micrograd.engine.Value;
import org.junit.jupiter.api.*;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("MLP 神經網路測試")
class MLPTest {

    @Test
    @DisplayName("MLP 參數數量正確")
    void testParameterCount() {
        // 3 → 4 → 1
        // Layer 1: 4 neurons × (3 weights + 1 bias) = 16
        // Layer 2: 1 neuron × (4 weights + 1 bias) = 5
        // Total: 21
        MLP mlp = new MLP(Arrays.asList(3, 4, 1));
        assertEquals(21, mlp.numParameters());
    }

    @Test
    @DisplayName("前向傳播輸出形狀正確")
    void testForwardShape() {
        MLP mlp = new MLP(Arrays.asList(3, 4, 2));
        List<Value> input = Arrays.asList(
                new Value(1.0),
                new Value(2.0),
                new Value(3.0)
        );
        List<Value> output = mlp.forward(input);
        assertEquals(2, output.size());
    }

    @Test
    @DisplayName("訓練後損失下降")
    void testTrainingReducesLoss() {
        Random rng = new Random(42);
        MLP mlp = new MLP(Arrays.asList(2, 4, 1), rng);

        // 簡單資料
        double[][] xs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[] ys = {0, 1, 1, 0}; // XOR

        // 計算初始損失
        double initialLoss = computeLoss(mlp, xs, ys);

        // 訓練 100 步
        for (int i = 0; i < 100; i++) {
            mlp.zeroGrad();
            Value loss = computeLossValue(mlp, xs, ys);
            loss.backward();
            for (Value p : mlp.parameters()) {
                p.setData(p.getData() - 0.1 * p.getGrad());
            }
        }

        // 計算最終損失
        double finalLoss = computeLoss(mlp, xs, ys);

        assertTrue(finalLoss < initialLoss,
                "損失應該下降: " + initialLoss + " -> " + finalLoss);
    }

    private double computeLoss(MLP mlp, double[][] xs, double[] ys) {
        return computeLossValue(mlp, xs, ys).getData();
    }

    private Value computeLossValue(MLP mlp, double[][] xs, double[] ys) {
        Value loss = new Value(0);
        for (int i = 0; i < xs.length; i++) {
            List<Value> input = new ArrayList<>();
            for (double x : xs[i]) {
                input.add(new Value(x));
            }
            Value pred = mlp.forwardSingle(input);
            Value diff = pred.sub(ys[i]);
            loss = loss.add(diff.pow(2));
        }
        return loss;
    }
}