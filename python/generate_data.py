import pandas as pd
import numpy as np
import os

def generate_synthetic_l2_data(num_snapshots=10000, n_levels=10, output_path="gnn_lob/data/l2_data.csv"):
    """
    Generates synthetic L2 LOB snapshots.
    Columns: timestamp, mid_price, bid_price_1, bid_vol_1, ..., ask_price_1, ask_vol_1, ...
    """
    np.random.seed(42)
    
    data = []
    mid_price = 100.0
    
    for i in range(num_snapshots):
        # random walk for mid price
        mid_price += np.random.normal(0, 0.1)
        
        # spread
        spread = 0.02 + np.random.exponential(0.01)
        
        best_bid = mid_price - spread/2
        best_ask = mid_price + spread/2
        
        snapshot = {
            'timestamp': i,
            'mid_price': mid_price
        }
        
        # Bid levels
        for level in range(n_levels):
            snapshot[f'bid_price_{level+1}'] = best_bid - level * 0.01
            snapshot[f'bid_vol_{level+1}'] = np.random.randint(100, 1000)
            
        # Ask levels
        for level in range(n_levels):
            snapshot[f'ask_price_{level+1}'] = best_ask + level * 0.01
            snapshot[f'ask_vol_{level+1}'] = np.random.randint(100, 1000)
            
        data.append(snapshot)
        
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_snapshots} snapshots at {output_path}")

if __name__ == "__main__":
    generate_synthetic_l2_data()
