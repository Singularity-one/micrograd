package com.micrograd.nn;

import com.micrograd.engine.Value;
import java.util.*;

public class Neuron implements Module {

    private final List<Value> w;
    private final Value b;
    private final boolean nonlin;

    public Neuron(int nin, boolean nonlin, Random rng) {
        this.w = new ArrayList<>();
        for (int i = 0; i < nin; i++) {
            this.w.add(new Value(rng.nextDouble() * 2 - 1));
        }
        this.b = new Value(0);
        this.nonlin = nonlin;
    }

    public Neuron(int nin, boolean nonlin) {
        this(nin, nonlin, new Random());
    }

    public Neuron(int nin) {
        this(nin, true);
    }

    public Value forward(List<Value> x) {
        // w · x + b
        Value act = b;
        for (int i = 0; i < w.size(); i++) {
            act = act.add(w.get(i).mul(x.get(i)));
        }
        return nonlin ? act.tanh() : act;
    }

    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>(w);
        params.add(b);
        return params;
    }

    @Override
    public String toString() {
        return String.format("%s Neuron(%d)",
                nonlin ? "tanh" : "Linear", w.size());
    }
}