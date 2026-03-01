#include "model_wrapper.hpp"
#include <iostream>

ModelWrapper::ModelWrapper(const std::string& model_path) {
    try {
        module_ = torch::jit::load(model_path);
        module_.eval();
        is_gpu_ = torch::cuda::is_available();
        if (is_gpu_) {
            module_.to(torch::kCUDA);
            std::cout << "Model loaded to GPU." << std::endl;
        } else {
            std::cout << "Model loaded to CPU." << std::endl;
        }
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.msg() << std::endl;
    }
}

torch::Tensor ModelWrapper::predict(const torch::Tensor& x, const torch::Tensor& edge_index) {
    std::vector<torch::jit::IValue> inputs;
    
    if (is_gpu_) {
        inputs.push_back(x.to(torch::kCUDA));
        inputs.push_back(edge_index.to(torch::kCUDA));
    } else {
        inputs.push_back(x);
        inputs.push_back(edge_index);
    }

    // Forward pass
    // For GNNs exported via trace, we pass x and edge_index
    auto output = module_.forward(inputs).toTensor();
    return output;
}
