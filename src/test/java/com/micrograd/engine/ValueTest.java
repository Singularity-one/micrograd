package com.micrograd.engine;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Value 自動微分引擎測試")
class ValueTest {

    private static final double TOL = 1e-6;
    private static final double H = 1e-5;

    @Nested
    @DisplayName("前向傳播測試")
    class ForwardPassTests {

        @Test
        @DisplayName("加法")
        void testAdd() {
            Value a = new Value(2.0);
            Value b = new Value(3.0);
            assertEquals(5.0, a.add(b).getData(), TOL);
        }

        @Test
        @DisplayName("乘法")
        void testMul() {
            Value a = new Value(2.0);
            Value b = new Value(3.0);
            assertEquals(6.0, a.mul(b).getData(), TOL);
        }

        @Test
        @DisplayName("冪次")
        void testPow() {
            Value a = new Value(2.0);
            assertEquals(8.0, a.pow(3).getData(), TOL);
        }

        @Test
        @DisplayName("tanh")
        void testTanh() {
            Value a = new Value(0.0);
            assertEquals(0.0, a.tanh().getData(), TOL);

            Value b = new Value(1.0);
            assertEquals(Math.tanh(1.0), b.tanh().getData(), TOL);
        }

        @Test
        @DisplayName("ReLU")
        void testRelu() {
            assertEquals(0.0, new Value(-5.0).relu().getData(), TOL);
            assertEquals(5.0, new Value(5.0).relu().getData(), TOL);
        }

        @Test
        @DisplayName("減法")
        void testSub() {
            Value a = new Value(5.0);
            Value b = new Value(3.0);
            assertEquals(2.0, a.sub(b).getData(), TOL);
        }

        @Test
        @DisplayName("除法")
        void testDiv() {
            Value a = new Value(6.0);
            Value b = new Value(2.0);
            assertEquals(3.0, a.div(b).getData(), TOL);
        }
    }

    @Nested
    @DisplayName("反向傳播梯度測試")
    class BackwardPassTests {

        @Test
        @DisplayName("加法梯度")
        void testAddGradient() {
            Value a = new Value(2.0);
            Value b = new Value(3.0);
            Value c = a.add(b);
            c.backward();

            assertEquals(1.0, a.getGrad(), TOL);
            assertEquals(1.0, b.getGrad(), TOL);
        }

        @Test
        @DisplayName("乘法梯度")
        void testMulGradient() {
            Value a = new Value(2.0);
            Value b = new Value(-3.0);
            Value c = a.mul(b);
            c.backward();

            assertEquals(-3.0, a.getGrad(), TOL); // dc/da = b
            assertEquals(2.0, b.getGrad(), TOL);  // dc/db = a
        }

        @Test
        @DisplayName("冪次梯度")
        void testPowGradient() {
            Value a = new Value(3.0);
            Value b = a.pow(2);
            b.backward();

            // d(x^2)/dx = 2x = 6
            assertEquals(6.0, a.getGrad(), TOL);
        }

        @Test
        @DisplayName("tanh 梯度")
        void testTanhGradient() {
            Value x = new Value(0.5);
            Value y = x.tanh();
            y.backward();

            double t = Math.tanh(0.5);
            double expected = 1 - t * t;
            assertEquals(expected, x.getGrad(), TOL);
        }

        @Test
        @DisplayName("ReLU 梯度")
        void testReluGradient() {
            Value a = new Value(5.0);
            Value b = a.relu();
            b.backward();
            assertEquals(1.0, a.getGrad(), TOL);

            Value c = new Value(-5.0);
            Value d = c.relu();
            d.backward();
            assertEquals(0.0, c.getGrad(), TOL);
        }
    }

    @Nested
    @DisplayName("數值梯度檢查")
    class NumericalGradientCheck {

        @Test
        @DisplayName("複合表達式梯度檢查")
        void testComplexExpression() {
            // f(a,b,c) = tanh(a * b + c)
            Value a = new Value(2.0);
            Value b = new Value(-3.0);
            Value c = new Value(10.0);

            Value out = a.mul(b).add(c).tanh();
            out.backward();

            // 數值梯度 for a
            double f1 = Math.tanh((2.0 + H) * (-3.0) + 10.0);
            double f2 = Math.tanh((2.0 - H) * (-3.0) + 10.0);
            double numericalGradA = (f1 - f2) / (2 * H);

            assertEquals(numericalGradA, a.getGrad(), 1e-4);
        }
    }

    @Nested
    @DisplayName("邊界情況")
    class EdgeCases {

        @Test
        @DisplayName("變數重複使用：a + a")
        void testVariableReuse() {
            Value a = new Value(3.0);
            Value b = a.add(a);
            b.backward();

            assertEquals(2.0, a.getGrad(), TOL);
        }

        @Test
        @DisplayName("更複雜的重複使用：a * a")
        void testSquare() {
            Value a = new Value(3.0);
            Value b = a.mul(a);
            b.backward();

            // d(a*a)/da = 2a = 6
            assertEquals(6.0, a.getGrad(), TOL);
        }

        @Test
        @DisplayName("鏈式重複使用")
        void testChainedReuse() {
            Value a = new Value(2.0);
            Value b = a.add(a);     // 2a
            Value c = b.mul(a);     // 2a * a = 2a²
            c.backward();

            // dc/da = 4a = 8
            assertEquals(8.0, a.getGrad(), TOL);
        }
    }
}