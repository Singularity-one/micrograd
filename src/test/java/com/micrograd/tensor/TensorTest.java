package com.micrograd.tensor;

import org.junit.jupiter.api.*;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Tensor 張量測試")
class TensorTest {

    private static final double TOL = 1e-6;

    @Nested
    @DisplayName("建構與基本操作")
    class ConstructionTests {

        @Test
        @DisplayName("建立零張量")
        void testZeros() {
            Tensor t = Tensor.zeros(3, 4);
            assertEquals(3, t.getRows());
            assertEquals(4, t.getCols());
            assertEquals(0.0, t.sum(), TOL);
        }

        @Test
        @DisplayName("建立一張量")
        void testOnes() {
            Tensor t = Tensor.ones(2, 3);
            assertEquals(6.0, t.sum(), TOL);
        }

        @Test
        @DisplayName("從陣列建立")
        void testFromArray() {
            double[][] arr = {{1, 2, 3}, {4, 5, 6}};
            Tensor t = Tensor.fromArray(arr);
            assertEquals(2, t.getRows());
            assertEquals(3, t.getCols());
            assertEquals(1.0, t.get(0, 0), TOL);
            assertEquals(6.0, t.get(1, 2), TOL);
        }

        @Test
        @DisplayName("One-hot 編碼")
        void testOneHot() {
            Tensor t = Tensor.oneHot(3, 5);
            assertEquals(1, t.getRows());
            assertEquals(5, t.getCols());
            assertEquals(1.0, t.get(0, 3), TOL);
            assertEquals(1.0, t.sum(), TOL);
        }
    }

    @Nested
    @DisplayName("數學運算")
    class MathTests {

        @Test
        @DisplayName("加法")
        void testAdd() {
            Tensor a = Tensor.fromArray(new double[][]{{1, 2}, {3, 4}});
            Tensor b = Tensor.fromArray(new double[][]{{5, 6}, {7, 8}});
            Tensor c = a.add(b);
            assertEquals(6.0, c.get(0, 0), TOL);
            assertEquals(12.0, c.get(1, 1), TOL);
        }

        @Test
        @DisplayName("乘法（元素對元素）")
        void testMul() {
            Tensor a = Tensor.fromArray(new double[][]{{1, 2}, {3, 4}});
            Tensor b = Tensor.fromArray(new double[][]{{2, 3}, {4, 5}});
            Tensor c = a.mul(b);
            assertEquals(2.0, c.get(0, 0), TOL);
            assertEquals(20.0, c.get(1, 1), TOL);
        }

        @Test
        @DisplayName("矩陣乘法")
        void testMatmul() {
            Tensor a = Tensor.fromArray(new double[][]{{1, 2}, {3, 4}});
            Tensor b = Tensor.fromArray(new double[][]{{5, 6}, {7, 8}});
            Tensor c = a.matmul(b);
            // [1,2] @ [5,6] = 1*5 + 2*7 = 19
            //         [7,8]   1*6 + 2*8 = 22
            assertEquals(19.0, c.get(0, 0), TOL);
            assertEquals(22.0, c.get(0, 1), TOL);
        }

        @Test
        @DisplayName("轉置")
        void testTranspose() {
            Tensor a = Tensor.fromArray(new double[][]{{1, 2, 3}, {4, 5, 6}});
            Tensor t = a.transpose();
            assertEquals(3, t.getRows());
            assertEquals(2, t.getCols());
            assertEquals(1.0, t.get(0, 0), TOL);
            assertEquals(4.0, t.get(0, 1), TOL);
            assertEquals(3.0, t.get(2, 0), TOL);
        }
    }

    @Nested
    @DisplayName("聚合運算")
    class AggregationTests {

        @Test
        @DisplayName("總和")
        void testSum() {
            Tensor t = Tensor.fromArray(new double[][]{{1, 2}, {3, 4}});
            assertEquals(10.0, t.sum(), TOL);
        }

        @Test
        @DisplayName("沿軸加總 - axis=0")
        void testSumAxis0() {
            Tensor t = Tensor.fromArray(new double[][]{{1, 2}, {3, 4}});
            Tensor s = t.sum(0);
            assertEquals(1, s.getRows());
            assertEquals(2, s.getCols());
            assertEquals(4.0, s.get(0, 0), TOL);  // 1+3
            assertEquals(6.0, s.get(0, 1), TOL);  // 2+4
        }

        @Test
        @DisplayName("沿軸加總 - axis=1")
        void testSumAxis1() {
            Tensor t = Tensor.fromArray(new double[][]{{1, 2}, {3, 4}});
            Tensor s = t.sum(1);
            assertEquals(2, s.getRows());
            assertEquals(1, s.getCols());
            assertEquals(3.0, s.get(0, 0), TOL);  // 1+2
            assertEquals(7.0, s.get(1, 0), TOL);  // 3+4
        }

        @Test
        @DisplayName("平均")
        void testMean() {
            Tensor t = Tensor.fromArray(new double[][]{{1, 2}, {3, 4}});
            assertEquals(2.5, t.mean(), TOL);
        }

        @Test
        @DisplayName("Argmax")
        void testArgmax() {
            Tensor t = Tensor.fromArray(new double[][]{{0.1, 0.5, 0.3, 0.1}});
            assertEquals(1, t.argmax());
        }
    }

    @Nested
    @DisplayName("元素運算")
    class ElementWiseTests {

        @Test
        @DisplayName("指數")
        void testExp() {
            Tensor t = Tensor.fromArray(new double[][]{{0, 1}});
            Tensor e = t.exp();
            assertEquals(1.0, e.get(0, 0), TOL);
            assertEquals(Math.E, e.get(0, 1), TOL);
        }

        @Test
        @DisplayName("對數")
        void testLog() {
            Tensor t = Tensor.fromArray(new double[][]{{1, Math.E}});
            Tensor l = t.log();
            assertEquals(0.0, l.get(0, 0), TOL);
            assertEquals(1.0, l.get(0, 1), TOL);
        }
    }
}

