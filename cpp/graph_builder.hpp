#pragma once
#include <torch/torch.h>
#include <vector>

class GraphBuilder {
public:
    GraphBuilder(int n_levels = 10);
    
    // Returns {x, edge_index}
    std::pair<torch::Tensor, torch::Tensor> create_graph(const std::vector<double>& snapshot);

private:
    int n_levels_;
    int num_nodes_;
    torch::Tensor edge_index_;
    
    void build_edge_index();
};
