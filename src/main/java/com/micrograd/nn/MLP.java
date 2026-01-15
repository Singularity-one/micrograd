package com.micrograd.nn;

import com.micrograd.engine.Value;
import java.util.*;

public class MLP implements Module {

    private final List<Layer> layers;

    public MLP(List<Integer> sizes, Random rng) {
        this.layers = new ArrayList<>();
        for (int i = 0; i < sizes.size() - 1; i++) {
            boolean nonlin = (i != sizes.size() - 2);
            layers.add(new Layer(sizes.get(i), sizes.get(i + 1), nonlin, rng));
        }
    }

    public MLP(List<Integer> sizes) {
        this(sizes, new Random());
    }

    public List<Value> forward(List<Value> x) {
        for (Layer layer : layers) {
            x = layer.forward(x);
        }
        return x;
    }

    public Value forwardSingle(List<Value> x) {
        return forward(x).get(0);
    }

    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.parameters());
        }
        return params;
    }

    @Override
    public String toString() {
        return String.format("MLP of [%s]", layers);
    }
}