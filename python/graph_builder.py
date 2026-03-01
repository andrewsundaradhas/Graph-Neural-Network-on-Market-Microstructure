import torch
from torch_geometric.data import Data
import numpy as np

class LOBGraphBuilder:
    def __init__(self, n_levels=10):
        self.n_levels = n_levels
        self.num_nodes = 2 * n_levels
        self.edge_index = self._build_edge_index()

    def _build_edge_index(self):
        edges = []
        # Intra-side edges (Bid side 0-9)
        for i in range(self.n_levels - 1):
            edges.append([i, i+1])
            edges.append([i+1, i])
        
        # Intra-side edges (Ask side 10-19)
        for i in range(self.n_levels, 2 * self.n_levels - 1):
            edges.append([i, i+1])
            edges.append([i+1, i])
            
        # Cross-side edges (bid_i <-> ask_i)
        for i in range(self.n_levels):
            bid_idx = i
            ask_idx = i + self.n_levels
            edges.append([bid_idx, ask_idx])
            edges.append([ask_idx, bid_idx])
            
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def create_graph(self, snapshot_row, label):
        """
        snapshot_row: array of numbers [bid_p1..p10, bid_v1..v10, ask_p1..p10, ask_v1..v10]
        """
        n = self.n_levels
        bid_prices = snapshot_row[0:n]
        bid_vols = snapshot_row[n:2*n]
        ask_prices = snapshot_row[2*n:3*n]
        ask_vols = snapshot_row[3*n:4*n]
        
        mid_price = (bid_prices[0] + ask_prices[0]) / 2.0
        
        node_features = []
        
        # Bid Nodes
        for i in range(n):
            node_features.append([
                (bid_prices[i] - mid_price) / mid_price, # Relative price
                bid_vols[i] / 1000.0,                    # Normalized volume
                0.0,                                      # Side (Bid=0)
                bid_vols[i] / (bid_vols[i] + ask_vols[i] + 1e-6) # Imbalance
            ])
            
        # Ask Nodes
        for i in range(n):
            node_features.append([
                (ask_prices[i] - mid_price) / mid_price, # Relative price
                ask_vols[i] / 1000.0,                    # Normalized volume
                1.0,                                      # Side (Ask=1)
                ask_vols[i] / (bid_vols[i] + ask_vols[i] + 1e-6) # Imbalance
            ])
            
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)
        
        return Data(x=x, edge_index=self.edge_index, y=y)

if __name__ == "__main__":
    builder = LOBGraphBuilder(n_levels=10)
    # Dummy snapshot: 10 bid prices, 10 bid vols, 10 ask prices, 10 ask vols
    dummy_snapshot = np.concatenate([
        np.linspace(99, 100, 10), # bid prices
        np.random.randint(100, 500, 10), # bid vols
        np.linspace(100.1, 101.1, 10), # ask prices
        np.random.randint(100, 500, 10) # ask vols
    ])
    graph = builder.create_graph(dummy_snapshot, 1)
    print(graph)
    print("Node features:", graph.x.shape)
    print("Edge index:", graph.edge_index.shape)
