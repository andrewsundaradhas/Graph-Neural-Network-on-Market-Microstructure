import torch
from torch_geometric.loader import DataLoader
from data_parser import LOBDataParser
from graph_builder import LOBGraphBuilder
from model import LOB_GNN
import os
from tqdm import tqdm

def train():
    # 1. Load and Prepare Data
    parser = LOBDataParser(n_levels=10, horizon=10)
    df = parser.load_and_label("gnn_lob/data/l2_data.csv")
    
    # For simplicity, we just take the raw columns and let GraphBuilder handle normalization
    # Actually, GraphBuilder expects the raw snapshot.
    
    bid_prices = [f'bid_price_{i+1}' for i in range(10)]
    bid_vols = [f'bid_vol_{i+1}' for i in range(10)]
    ask_prices = [f'ask_price_{i+1}' for i in range(10)]
    ask_vols = [f'ask_vol_{i+1}' for i in range(10)]
    
    feature_cols = bid_prices + bid_vols + ask_prices + ask_vols
    X_raw = df[feature_cols].values
    y = df['y'].values
    
    builder = LOBGraphBuilder(n_levels=10)
    dataset = []
    print("Constructing graphs...")
    for i in tqdm(range(len(X_raw))):
        data = builder.create_graph(X_raw[i], y[i])
        dataset.append(data)
    
    # 2. Split Data
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 3. Model, Optimizer, Loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LOB_GNN(num_node_features=4, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 4. Training Loop
    epochs = 10
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
        
        acc = correct / len(val_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}")
        
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), "gnn_lob/models/gnn_model_state.pth")
            print("Model saved.")

if __name__ == "__main__":
    train()
