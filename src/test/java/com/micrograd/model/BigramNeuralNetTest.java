package com.micrograd.model;
import com.micrograd.data.Vocabulary;
import org.junit.jupiter.api.*;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("BigramNeuralNet 神經網路模型測試")
class BigramNeuralNetTest {

    private Vocabulary vocab;

    @BeforeEach
    void setUp() {
        vocab = new Vocabulary();
    }

    @Test
    @DisplayName("訓練後損失應下降")
    void testTrainingReducesLoss() {
        Random rng = new Random(42);
        BigramNeuralNet model = new BigramNeuralNet(vocab, 50.0, 50, rng);

        // 簡單的訓練資料
        int[][] bigrams = {
                {0, 1, 2, 0, 1, 3},
                {1, 2, 0, 1, 3, 0}
        };

        // 訓練前損失
        double initialLoss = model.loss(bigrams);

        // 訓練
        model.train(bigrams);

        // 訓練後損失
        double finalLoss = model.loss(bigrams);

        assertTrue(finalLoss < initialLoss,
                "Loss should decrease: " + initialLoss + " -> " + finalLoss);
    }

    @Test
    @DisplayName("生成的名字應只包含有效字元")
    void testSampleValidChars() {
        Random rng = new Random(42);
        BigramNeuralNet model = new BigramNeuralNet(vocab, 50.0, 20, rng);

        int[][] bigrams = {
                {0, 1, 2, 0, 3, 4, 5},
                {1, 2, 0, 3, 4, 5, 0}
        };

        model.train(bigrams);

        for (int i = 0; i < 10; i++) {
            String name = model.sample(rng);
            for (char c : name.toCharArray()) {
                assertTrue(vocab.contains(c), "Invalid char: " + c);
                assertNotEquals('.', c, "Name should not contain '.'");
            }
        }
    }

    @Test
    @DisplayName("權重矩陣形狀應為 27x27")
    void testWeightShape() {
        Random rng = new Random(42);
        BigramNeuralNet model = new BigramNeuralNet(vocab, 50.0, 10, rng);

        assertEquals(27, model.getWeights().getRows());
        assertEquals(27, model.getWeights().getCols());
    }

    @Test
    @DisplayName("機率矩陣每列應加總為 1")
    void testProbsNormalized() {
        Random rng = new Random(42);
        BigramNeuralNet model = new BigramNeuralNet(vocab, 50.0, 10, rng);

        int[][] bigrams = {{0, 1}, {1, 0}};
        model.train(bigrams);

        var probs = model.getProbs();
        for (int i = 0; i < vocab.size(); i++) {
            double rowSum = probs.getRow(i).sum();
            assertEquals(1.0, rowSum, 1e-5, "Row " + i + " should sum to 1");
        }
    }
}