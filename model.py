import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(input_size, 128, device=self.device)
        self.fc2 = nn.Linear(128, 64, device=self.device)
        self.fc3 = nn.Linear(64, output_size, device=self.device)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TransformerQNetwork(nn.Module):
    def __init__(self, state_size, action_size, feature_size, num_heads, num_layers, hidden_size=256):
        super(TransformerQNetwork, self).__init__()

        self.feature_size = feature_size  # The size of the expected features in the state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure the feature size is divisible by the number of heads
        assert self.feature_size % num_heads == 0, "feature_size must be divisible by num_heads"
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_size, 
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, 
            num_layers=num_layers, 
            enable_nested_tensor=False
        )

        # Define the rest of the Q-network
        # self.fc1 = nn.Linear(state_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        # only encoder
        state_features = self.transformer_encoder(state).to(self.device)
        # print(f'state_features: {state_features}')
        # print(f'Shape of state_features: {state_features.shape}')

        # no effect as it will be a fixed size (batch_size, state_size)
        # q_input = state_features.flatten(start_dim=1)  # Flatten all features
        q_input = state_features
        # print(f'Shape of q_input: {q_input.shape}')
        # print(q_input.shape)
        
        # x = F.relu(self.fc1(q_input))
        # return self.fc2(x)
        return q_input