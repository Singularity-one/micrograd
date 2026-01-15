package com.micrograd.nn;

import com.micrograd.engine.Value;
import java.util.List;

public interface Module {

    List<Value> parameters();

    default void zeroGrad() {
        for (Value p : parameters()) {
            p.setGrad(0.0);
        }
    }

    default int numParameters() {
        return parameters().size();
    }
}