import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class LOBDataParser:
    def __init__(self, n_levels=10, horizon=50):
        self.n_levels = n_levels
        self.horizon = horizon
        self.scaler = StandardScaler()

    def load_and_label(self, file_path):
        df = pd.read_csv(file_path)
        
        # Calculate future mid-price return
        # mid_t+delta - mid_t
        df['label'] = df['mid_price'].shift(-self.horizon) - df['mid_price']
        
        # Drop rows where we don't have enough future data for labeling
        df.dropna(subset=['label'], inplace=True)
        
        # Classification labels: 1 for Up, 2 for Down, 0 for Neutral (roughly)
        # For simplicity, we use binary for now or sign
        def classify(ret):
            threshold = 0.01 # Volatility threshold for "neutral"
            if ret > threshold:
                return 1 # Up
            elif ret < -threshold:
                return 2 # Down
            else:
                return 0 # Neutral
        
        df['y'] = df['label'].apply(classify)
        
        return df

    def get_features(self, df):
        # Extract price and volume columns
        bid_prices = [f'bid_price_{i+1}' for i in range(self.n_levels)]
        bid_vols = [f'bid_vol_{i+1}' for i in range(self.n_levels)]
        ask_prices = [f'ask_price_{i+1}' for i in range(self.n_levels)]
        ask_vols = [f'ask_vol_{i+1}' for i in range(self.n_levels)]
        
        features = df[bid_prices + bid_vols + ask_prices + ask_vols].values
        # Fit scaler on training data (simplified for this task)
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled, df['y'].values

if __name__ == "__main__":
    parser = LOBDataParser(n_levels=10, horizon=10)
    df = parser.load_and_label("gnn_lob/data/l2_data.csv")
    X, y = parser.get_features(df)
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Label distribution: \n{pd.Series(y).value_counts()}")
