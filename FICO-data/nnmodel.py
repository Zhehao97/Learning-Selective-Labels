import torch.nn as nn

# Define a simple neural-network model
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.fc(x))
        return output

   
# Neural Network Classifier
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

class ImprovedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[64, 32], dropout_rate=0.3):
        """
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(ImprovedClassifier, self).__init__()
        
        # Create a list to store layers
        layers = []
        
        # Input layer to first hidden layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # Activation function
            layers.append(nn.BatchNorm1d(hidden_dim))  # Batch normalization
            layers.append(nn.Dropout(dropout_rate))  # Dropout
            prev_dim = hidden_dim

        # Final layer to output classes
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Define the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[128, 64], dropout_prob=0.5):
        """
        Enhanced neural network classifier with multiple layers and regularization.
        
        Args:
            input_dim (int): Number of input features.
            num_classes (int): Number of output classes.
            hidden_dims (list of int): Sizes of hidden layers.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(EnhancedClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))  # Batch normalization
            layers.append(nn.Dropout(dropout_prob))    # Dropout for regularization
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        # Define the model as a sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)