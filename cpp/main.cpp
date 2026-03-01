#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include "graph_builder.hpp"
#include "model_wrapper.hpp"
#include "strategy.hpp"

// Mock feed for LOB data
std::vector<double> get_mock_snapshot(int n_levels) {
    std::vector<double> snapshot;
    // Prices
    for (int i = 0; i < n_levels; ++i) snapshot.push_back(100.0 - i * 0.01); // Bid P
    // Volumes
    for (int i = 0; i < n_levels; ++i) snapshot.push_back(500.0);           // Bid V
    // Prices
    for (int i = 0; i < n_levels; ++i) snapshot.push_back(100.01 + i * 0.01); // Ask P
    // Volumes
    for (int i = 0; i < n_levels; ++i) snapshot.push_back(500.0);           // Ask V
    return snapshot;
}

int main(int argc, char* argv[]) {
    std::string model_path = "../models/gnn_model_traced.pt";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "Starting GNN LOB Inference Engine..." << std::endl;
    std::cout << "Loading model from: " << model_path << std::endl;

    GraphBuilder builder(10);
    ModelWrapper model(model_path);
    StrategyEngine strategy(0.4); // Local testing threshold

    // Warm up
    auto dummy_snapshot = get_mock_snapshot(10);
    auto [dummy_x, dummy_edge] = builder.create_graph(dummy_snapshot);
    model.predict(dummy_x, dummy_edge);

    std::cout << "Inference loop started." << std::endl;

    for (int i = 0; i < 10; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        // 1. Get Market Data
        auto snapshot = get_mock_snapshot(10);

        // 2. Build Graph
        auto [x, edge_index] = builder.create_graph(snapshot);

        // 3. Inference
        auto prediction = model.predict(x, edge_index);

        // 4. Strategy Decision
        std::string signal = strategy.decide(prediction);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "[Iteration " << i << "] Signal: " << signal 
                  << " | Latency: " << duration.count() << " us" << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}
