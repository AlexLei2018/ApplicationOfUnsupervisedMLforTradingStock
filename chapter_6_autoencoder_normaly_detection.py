import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Ensure all required libraries are imported
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1. Data Preparation ---
# Ensure market_features and training_period_end are already defined in the environment.
# For this code block to be self-contained, we will rerun the data preparation logic here.
print("Preparing data for the Autoencoder...")
spy_data_ae = yf.download('SPY', start='2018-01-01', end='2021-12-31', auto_adjust=True, progress=False)
if spy_data_ae.empty:
    print("Error: Failed to download SPY data.")
else:
    returns_ae = spy_data_ae['Close'].pct_change()
    market_features_ae = pd.DataFrame(index=returns_ae.index)
    market_features_ae['Return'] = returns_ae
    market_features_ae['Volatility_20D'] = returns_ae.rolling(20).std() * np.sqrt(252)
    market_features_ae.dropna(inplace=True)

    training_period_end = '2020-01-31'
    train_data_ae = market_features_ae.loc[:training_period_end]

    scaler_ae = StandardScaler().fit(train_data_ae)
    
    scaled_features_ae = scaler_ae.transform(market_features_ae)
    train_tensor = torch.tensor(scaler_ae.transform(train_data_ae), dtype=torch.float32)
    features_tensor = torch.tensor(scaled_features_ae, dtype=torch.float32)
    
    train_dataset = TensorDataset(train_tensor, train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # --- 2. [REVISED] Define a more robust Autoencoder model ---
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim=2):
            super(Autoencoder, self).__init__()
            # Encoder: Progressively compresses the input dimension to the bottleneck dimension
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(True),
                nn.Linear(16, 8),
                nn.ReLU(True),
                nn.Linear(8, encoding_dim) # Bottleneck layer
            )
            # Decoder: Progressively decompresses from the bottleneck dimension back to the original dimension
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 8),
                nn.ReLU(True),
                nn.Linear(8, 16),
                nn.ReLU(True),
                nn.Linear(16, input_dim) # Output layer
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        
        def encode(self, x):
            return self.encoder(x)

    input_dim = train_tensor.shape[1]
    model = Autoencoder(input_dim=input_dim, encoding_dim=2) # Assuming a bottleneck dimension of 2
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Print the model structure for inspection
    print("Autoencoder Model Structure:")
    print(model)

    # --- 3. Train the Model ---
    print("Training the Autoencoder to learn 'normal' patterns...")
    num_epochs = 100
    for epoch in range(num_epochs):
        for data in train_dataloader:
            inputs, _ = data
            # Forward pass
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, inputs)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("Model training complete.")

    # --- 4. Calculate Reconstruction Error for the Entire Time Series ---
    model.eval()
    with torch.no_grad():
        reconstructions = model(features_tensor)
        # Calculate MSE row by row
        reconstruction_errors = torch.mean((features_tensor - reconstructions)**2, axis=1)

    results_ae_df = market_features_ae.copy()
    results_ae_df['Reconstruction_Error'] = reconstruction_errors.numpy()

    # --- 5. Visualize the Results ---
    error_threshold = np.quantile(results_ae_df['Reconstruction_Error'].loc[train_data_ae.index], 0.99)
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()
    
    # Ensure indices are aligned for plotting
    price_plot = spy_data_ae['Close'].reindex(results_ae_df.index, method='ffill')

    ax1.plot(price_plot.index, price_plot.values, color='black', label='SPY Price')
    ax2.plot(results_ae_df.index, results_ae_df['Reconstruction_Error'], color='purple', alpha=0.6, label='Reconstruction Error')
    ax2.axhline(y=error_threshold, color='red', linestyle='--', label=f'99% Anomaly Threshold ({error_threshold:.3f})')
    
    ax1.set_title('Autoencoder Anomaly Detection via Reconstruction Error')
    ax1.set_ylabel('SPY Price')
    ax2.set_ylabel('Reconstruction Error', color='purple')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.show()
