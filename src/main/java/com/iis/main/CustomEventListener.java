package com.iis.main;

import java.util.EventListener;

public interface CustomEventListener extends EventListener {
    void updateRegressionScore(int iteration, double modelScore);
    void updateEpochValidationScore(int epoch, double score);
}
