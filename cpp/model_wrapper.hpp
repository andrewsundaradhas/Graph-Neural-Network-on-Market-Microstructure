#pragma once
#include <torch/script.h>
#include <memory>

class ModelWrapper {
public:
    ModelWrapper(const std::string& model_path);
    torch::Tensor predict(const torch::Tensor& x, const torch::Tensor& edge_index);

private:
    torch::jit::script::Module module_;
    bool is_gpu_;
};
