import torch
from model import LOB_GNN
import os

def export_model():
    # 1. Initialize model
    model = LOB_GNN(num_node_features=4, num_classes=3)
    
    # 2. Load state dict if exists
    state_dict_path = "gnn_lob/models/gnn_model_state.pth"
    if os.path.exists(state_dict_path):
        model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
        print(f"Loaded model state from {state_dict_path}")
    else:
        print("Warning: No model state found, exporting untrained model.")
        
    model.eval()
    
    # 3. Create dummy inputs for tracing or scripting
    # Scripting is generally better for GNNs
    try:
        scripted_model = torch.jit.script(model)
        output_path = "gnn_lob/models/gnn_model.pt"
        scripted_model.save(output_path)
        print(f"Successfully exported scripted model to {output_path}")
    except Exception as e:
        print(f"Error during scripting: {e}")
        print("Falling back to tracing (might lose dynamic control flow but easier for simple models).")
        # Dummy inputs
        x = torch.randn(20, 4)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        traced_model = torch.jit.trace(model, (x, edge_index))
        traced_model.save("gnn_lob/models/gnn_model_traced.pt")
        print("Exported traced model to gnn_lob/models/gnn_model_traced.pt")

if __name__ == "__main__":
    export_model()
