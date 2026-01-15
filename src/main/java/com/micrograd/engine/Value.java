package com.micrograd.engine;

import java.util.*;

public class Value {

    private double data;
    private double grad;
    private final Set<Value> prev;
    private final String op;
    private String label;
    private Runnable backward;

    // === 建構子 ===

    public Value(double data) {
        this(data, Collections.emptySet(), "");
    }

    public Value(double data, Set<Value> children, String op) {
        this.data = data;
        this.grad = 0.0;
        this.prev = new HashSet<>(children);  // 使用 HashSet
        this.op = op;
        this.label = "";
        this.backward = () -> {};
    }

    // === 輔助方法：建立子節點集合（允許重複）===

    private static Set<Value> childrenOf(Value... values) {
        Set<Value> set = new HashSet<>();
        Collections.addAll(set, values);
        return set;
    }

    // === 基本運算 ===

    public Value add(Value other) {
        Value out = new Value(
                this.data + other.data,
                childrenOf(this, other),  // ✅ 允許 this == other
                "+"
        );

        // 保存 this 和 other 的引用供 backward 使用
        Value self = this;

        out.backward = () -> {
            self.grad += out.grad;
            other.grad += out.grad;
        };

        return out;
    }

    public Value add(double other) {
        return this.add(new Value(other));
    }

    public Value mul(Value other) {
        Value out = new Value(
                this.data * other.data,
                childrenOf(this, other),  // ✅ 允許 this == other
                "*"
        );

        Value self = this;

        out.backward = () -> {
            self.grad += other.data * out.grad;
            other.grad += self.data * out.grad;
        };

        return out;
    }

    public Value mul(double other) {
        return this.mul(new Value(other));
    }

    public Value pow(double n) {
        Value out = new Value(
                Math.pow(this.data, n),
                childrenOf(this),
                "**" + n
        );

        Value self = this;

        out.backward = () -> {
            self.grad += n * Math.pow(self.data, n - 1) * out.grad;
        };

        return out;
    }

    // === 激活函數 ===

    public Value tanh() {
        double x = this.data;
        double t = (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);

        Value out = new Value(t, childrenOf(this), "tanh");

        Value self = this;

        out.backward = () -> {
            self.grad += (1 - t * t) * out.grad;
        };

        return out;
    }

    public Value relu() {
        Value out = new Value(
                Math.max(0, this.data),
                childrenOf(this),
                "ReLU"
        );

        Value self = this;

        out.backward = () -> {
            self.grad += (out.data > 0 ? 1.0 : 0.0) * out.grad;
        };

        return out;
    }

    public Value exp() {
        double t = Math.exp(this.data);

        Value out = new Value(t, childrenOf(this), "exp");

        Value self = this;

        out.backward = () -> {
            self.grad += t * out.grad;
        };

        return out;
    }

    // === 便利運算 ===

    public Value neg() {
        return this.mul(-1);
    }

    public Value sub(Value other) {
        return this.add(other.neg());
    }

    public Value sub(double other) {
        return this.add(-other);
    }

    public Value div(Value other) {
        return this.mul(other.pow(-1));
    }

    public Value div(double other) {
        return this.mul(1.0 / other);
    }

    // === 反向傳播 ===

    public void backward() {
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        buildTopo(this, topo, visited);

        this.grad = 1.0;

        Collections.reverse(topo);
        for (Value v : topo) {
            v.backward.run();
        }
    }

    private void buildTopo(Value v, List<Value> topo, Set<Value> visited) {
        if (!visited.contains(v)) {
            visited.add(v);
            for (Value child : v.prev) {
                buildTopo(child, topo, visited);
            }
            topo.add(v);
        }
    }

    // === Getter / Setter ===

    public double getData() {
        return data;
    }

    public void setData(double data) {
        this.data = data;
    }

    public double getGrad() {
        return grad;
    }

    public void setGrad(double grad) {
        this.grad = grad;
    }

    public Set<Value> getPrev() {
        return prev;
    }

    public String getOp() {
        return op;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return String.format("Value(data=%.4f, grad=%.4f)", data, grad);
    }
}