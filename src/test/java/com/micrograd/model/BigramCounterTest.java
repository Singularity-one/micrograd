package com.micrograd.model;

import com.micrograd.data.Vocabulary;
import com.micrograd.tensor.Tensor;
import org.junit.jupiter.api.*;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("BigramCounter 計數模型測試")
class BigramCounterTest {

    private Vocabulary vocab;
    private BigramCounter model;

    @BeforeEach
    void setUp() {
        vocab = new Vocabulary();
        model = new BigramCounter(vocab, 1.0);
    }

    @Test
    @DisplayName("訓練後機率分佈應正規化")
    void testNormalization() {
        // 模擬 "ab" 這個字的 bigrams: .->a, a->b, b->.
        int[][] bigrams = {
                {0, 1, 2},   // inputs: ., a, b
                {1, 2, 0}    // targets: a, b, .
        };

        model.train(bigrams);
        assertTrue(model.checkNormalization());
    }

    @Test
    @DisplayName("訓練後每列加總為 1")
    void testRowSums() {
        int[][] bigrams = {
                {0, 1, 2, 0, 3, 4},
                {1, 2, 0, 3, 4, 0}
        };

        model.train(bigrams);
        Tensor probs = model.getProbs();

        for (int i = 0; i < vocab.size(); i++) {
            double rowSum = probs.getRow(i).sum();
            assertEquals(1.0, rowSum, 1e-6, "Row " + i + " should sum to 1");
        }
    }

    @Test
    @DisplayName("損失應為正數")
    void testLossPositive() {
        int[][] bigrams = {
                {0, 1, 2},
                {1, 2, 0}
        };

        model.train(bigrams);
        double loss = model.loss(bigrams);
        assertTrue(loss > 0, "Loss should be positive");
    }

    @Test
    @DisplayName("生成的名字應只包含有效字元")
    void testSampleValidChars() {
        int[][] bigrams = {
                {0, 1, 2, 0, 3, 4},
                {1, 2, 0, 3, 4, 0}
        };

        model.train(bigrams);
        Random rng = new Random(42);

        for (int i = 0; i < 10; i++) {
            String name = model.sample(rng);
            for (char c : name.toCharArray()) {
                assertTrue(vocab.contains(c), "Invalid char: " + c);
                assertNotEquals('.', c, "Name should not contain '.'");
            }
        }
    }

    @Test
    @DisplayName("Smoothing 為 0 時可能產生零機率")
    void testNoSmoothing() {
        BigramCounter noSmoothModel = new BigramCounter(vocab, 0.0);

        int[][] bigrams = {
                {0, 1},
                {1, 0}
        };

        noSmoothModel.train(bigrams);
        Tensor probs = noSmoothModel.getProbs();

        // 未出現的 bigram 機率應為 0
        assertEquals(0.0, probs.get(2, 3), 1e-10);
    }
}
