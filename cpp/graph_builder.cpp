#include "graph_builder.hpp"
#include <iostream>

GraphBuilder::GraphBuilder(int n_levels) : n_levels_(n_levels) {
    num_nodes_ = 2 * n_levels_;
    build_edge_index();
}

void GraphBuilder::build_edge_index() {
    std::vector<int64_t> edges_src;
    std::vector<int64_t> edges_dst;

    // Intra-side edges (Bid side 0 to n_levels-1)
    for (int i = 0; i < n_levels_ - 1; ++i) {
        edges_src.push_back(i); edges_dst.push_back(i + 1);
        edges_src.push_back(i + 1); edges_dst.push_back(i);
    }

    // Intra-side edges (Ask side n_levels to 2n_levels-1)
    for (int i = n_levels_; i < 2 * n_levels_ - 1; ++i) {
        edges_src.push_back(i); edges_dst.push_back(i + 1);
        edges_src.push_back(i + 1); edges_dst.push_back(i);
    }

    // Cross-side edges (bid_i <-> ask_i)
    for (int i = 0; i < n_levels_; ++i) {
        int bid_idx = i;
        int ask_idx = i + n_levels_;
        edges_src.push_back(bid_idx); edges_dst.push_back(ask_idx);
        edges_src.push_back(ask_idx); edges_dst.push_back(bid_idx);
    }

    auto src_tensor = torch::tensor(edges_src, torch::kInt64);
    auto dst_tensor = torch::tensor(edges_dst, torch::kInt64);
    edge_index_ = torch::stack({src_tensor, dst_tensor}, 0);
}

std::pair<torch::Tensor, torch::Tensor> GraphBuilder::create_graph(const std::vector<double>& snapshot) {
    // snapshot layout: [bid_p1..p10, bid_v1..v10, ask_p1..p10, ask_v1..v10]
    const int n = n_levels_;
    double mid_price = (snapshot[0] + snapshot[2 * n]) / 2.0;

    auto x = torch::zeros({num_nodes_, 4}, torch::kFloat32);
    auto x_accessor = x.accessor<float, 2>();

    for (int i = 0; i < n; ++i) {
        double bid_p = snapshot[i];
        double bid_v = snapshot[n + i];
        double ask_p = snapshot[2 * n + i];
        double ask_v = snapshot[3 * n + i];

        // Bid Node
        x_accessor[i][0] = (float)((bid_p - mid_price) / mid_price);
        x_accessor[i][1] = (float)(bid_v / 1000.0);
        x_accessor[i][2] = 0.0f; // Side Bid
        x_accessor[i][3] = (float)(bid_v / (bid_v + ask_v + 1e-6));

        // Ask Node
        int ask_idx = i + n;
        x_accessor[ask_idx][0] = (float)((ask_p - mid_price) / mid_price);
        x_accessor[ask_idx][1] = (float)(ask_v / 1000.0);
        x_accessor[ask_idx][2] = 1.0f; // Side Ask
        x_accessor[ask_idx][3] = (float)(ask_v / (bid_v + ask_v + 1e-6));
    }

    return {x, edge_index_};
}
