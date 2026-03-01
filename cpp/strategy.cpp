#include "strategy.hpp"
#include <iostream>

StrategyEngine::StrategyEngine(double threshold) : threshold_(threshold) {}

std::string StrategyEngine::decide(const torch::Tensor& prediction) {
    // Prediction is expected to be [1, 3] tensor (Neutral, Up, Down)
    // Apply softmax
    auto probs = torch::softmax(prediction, 1);
    
    float prob_neutral = probs[0][0].item<float>();
    float prob_up = probs[0][1].item<float>();
    float prob_down = probs[0][2].item<float>();

    if (prob_up > threshold_) {
        return "BUY";
    } else if (prob_down > threshold_) {
        return "SELL";
    } else {
        return "NEUTRAL";
    }
}
