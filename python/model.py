import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class LOB_GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(LOB_GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, x, edge_index, batch=None):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        # 2. Readout layer (Global Mean Pooling)
        # If batch is None (inference), we assume it's one graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Final classifier
        x = self.fc(x)
        
        return x

if __name__ == "__main__":
    model = LOB_GNN(num_node_features=4, num_classes=3)
    print(model)
    
    # Test forward pass
    x = torch.randn(20, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    out = model(x, edge_index)
    print("Output shape:", out.shape)
