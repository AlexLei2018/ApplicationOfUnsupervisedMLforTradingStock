# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility of PyTorch experiments
torch.manual_seed(42)
np.random.seed(42)

#### 2. Define the Autoencoder Model

# We will define a simple, fully-connected (dense) autoencoder. 
# Its architecture will be: `Input Layer -> Hidden Layer 1 (Encoding) -> Bottleneck Layer -> Hidden Layer 2 (Decoding) -> Output Layer`.

# Define the Autoencoder network architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        """
        Initializes the Autoencoder model.

        Args:
            input_dim (int): The number of input features.
            bottleneck_dim (int): The dimension of the bottleneck layer (i.e., the number
                                  of nonlinear factors we want to extract).
        """
        super(Autoencoder, self).__init__()
        
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),      # Input layer -> Hidden layer 1
            nn.ReLU(),                     # Activation function
            nn.Linear(16, bottleneck_dim)  # Hidden layer 1 -> Bottleneck layer
        )
        
        # Decoder part
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 16), # Bottleneck layer -> Hidden layer 2
            nn.ReLU(),                     # Activation function
            nn.Linear(16, input_dim)       # Hidden layer 2 -> Output layer (reconstruction)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    def get_bottleneck_features(self, x):
        """A helper method to extract the bottleneck output after training."""
        return self.encoder(x)

#### 3. Train the Autoencoder

# Now, we will train our model using real stock feature data. 
# The model's goal is to make its output (`decoded`) as close as possible to the original input (`x`).


if __name__ == '__main__':
    # --- a. Data Preparation ---
    # Using the logic from previous chapters, we get and scale a feature snapshot.
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT', 'XOM']
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    
    price_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    returns = np.log(price_data / price_data.shift(1))
    
    # Build features
    volatility = returns.rolling(60).std() * np.sqrt(252)
    momentum = price_data.pct_change(63)
    
    feature_snapshot = pd.DataFrame({
        'volatility': volatility.iloc[-1],
        'momentum': momentum.iloc[-1]
    }).dropna()
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_snapshot)
    
    # Convert NumPy array to PyTorch Tensor
    features_tensor = torch.FloatTensor(scaled_features)
    
    # Create PyTorch Dataset and DataLoader
    dataset = TensorDataset(features_tensor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # --- b. Initialize Model and Training Parameters ---
    input_dimension = features_tensor.shape[1]  # Number of features (2 in this case)
    bottleneck_dimension = 1                    # We will try to compress the features into 1 dimension
    
    model = Autoencoder(input_dim=input_dimension, bottleneck_dim=bottleneck_dimension)
    criterion = nn.MSELoss()  # Loss function: Mean Squared Error to measure reconstruction difference
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer
    
    num_epochs = 100 # Number of training epochs

    # --- c. Training Loop ---
    print("Starting Autoencoder training...")
    for epoch in range(num_epochs):
        for data in dataloader:
            inputs = data[0]
            # 1. Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # 2. Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()        # Calculate gradients
            optimizer.step()       # Update weights
            
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print("Training complete!")
    
    # --- d. Extract and Interpret Results ---
    # Use the trained model to extract the bottleneck features
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # No need to calculate gradients for inference
        bottleneck_features = model.get_bottleneck_features(features_tensor)

    # Convert the results back to a pandas DataFrame for inspection
    nonlinear_factor = pd.DataFrame(
        bottleneck_features.numpy(),
        index=feature_snapshot.index,
        columns=[f'AE_Factor_{i+1}' for i in range(bottleneck_dimension)]
    )
    
    print("\n--- Original Feature Snapshot (Partial) ---")
    print(feature_snapshot.head())
    
    print(f"\n--- {bottleneck_dimension}-Dimensional Nonlinear Factor Extracted by Autoencoder ---")
    print(nonlinear_factor)