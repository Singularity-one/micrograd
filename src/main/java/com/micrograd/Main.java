package com.micrograd;
import com.micrograd.engine.Value;
import com.micrograd.nn.MLP;
import com.micrograd.viz.GraphVisualizer;

import java.util.*;

public class Main {

    public static void main(String[] args) {

        // === 範例 1：簡單表達式 ===
        Value a = new Value(2.0);
        a.setLabel("a");

        Value b = new Value(-3.0);
        b.setLabel("b");

        Value c = new Value(10.0);
        c.setLabel("c");

        Value e = a.mul(b);
        e.setLabel("e");

        Value d = e.add(c);
        d.setLabel("d");

        Value f = new Value(-2.0);
        f.setLabel("f");

        Value L = d.mul(f);
        L.setLabel("L");

        // 反向傳播
        L.backward();

        // 視覺化（自動開啟圖片）
        GraphVisualizer.draw(L, "example1_simple");


        // === 範例 2：神經元 ===
        Value x1 = new Value(2.0);
        x1.setLabel("x1");

        Value x2 = new Value(0.0);
        x2.setLabel("x2");

        Value w1 = new Value(-3.0);
        w1.setLabel("w1");

        Value w2 = new Value(1.0);
        w2.setLabel("w2");

        Value bias = new Value(6.8813735870195432);
        bias.setLabel("b");

        Value x1w1 = x1.mul(w1);
        x1w1.setLabel("x1*w1");

        Value x2w2 = x2.mul(w2);
        x2w2.setLabel("x2*w2");

        Value x1w1x2w2 = x1w1.add(x2w2);
        x1w1x2w2.setLabel("x1w1+x2w2");

        Value n = x1w1x2w2.add(bias);
        n.setLabel("n");

        Value o = n.tanh();
        o.setLabel("o");

        // 反向傳播
        o.backward();

        // 視覺化
        GraphVisualizer.draw(o, "example2_neuron");

        System.out.println("\n=== 神經元梯度 ===");
        System.out.println("x1.grad = " + x1.getGrad());
        System.out.println("x2.grad = " + x2.getGrad());
        System.out.println("w1.grad = " + w1.getGrad());
        System.out.println("w2.grad = " + w2.getGrad());
    }
}