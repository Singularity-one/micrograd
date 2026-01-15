package com.micrograd.nn;

import com.micrograd.engine.Value;
import java.util.*;

public class Layer implements Module {

    private final List<Neuron> neurons;

    public Layer(int nin, int nout, boolean nonlin, Random rng) {
        this.neurons = new ArrayList<>();
        for (int i = 0; i < nout; i++) {
            neurons.add(new Neuron(nin, nonlin, rng));
        }
    }

    public Layer(int nin, int nout, boolean nonlin) {
        this(nin, nout, nonlin, new Random());
    }

    public Layer(int nin, int nout) {
        this(nin, nout, true);
    }

    public List<Value> forward(List<Value> x) {
        List<Value> out = new ArrayList<>();
        for (Neuron n : neurons) {
            out.add(n.forward(x));
        }
        return out;
    }

    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (Neuron n : neurons) {
            params.addAll(n.parameters());
        }
        return params;
    }

    @Override
    public String toString() {
        return String.format("Layer of [%s]", neurons);
    }
}