#pragma once
#include <torch/torch.h>
#include <string>

class StrategyEngine {
public:
    StrategyEngine(double threshold = 0.6);
    std::string decide(const torch::Tensor& prediction);

private:
    double threshold_;
};
