import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim

class FullyConnectedNN(nn.Module):
    """ We build FullyConnectedNN for predicting phylogenetic distance. We use embeddings of synthetic sequences from MSA Transformer and ground truth patristic distances from FastTree. """
    def __init__(self, input_dim, hidden_layers = [8, 16, 8], output_dim=1, activation=nn.ReLU(), batch_size=32, dropout_prob=0.2):
        """ 
        Iniatize the classifier. 
            Args:
                hidden_layers (list): Sizes of the hidden layers.
                output_dim (int): Since we do regression output_dim=1.
                activation (callable): The activation function to apply after each hidden layer.
                batch_size (int): Batch_size.
                optimizer (torch.optim): Optimizer used for training. 
        """
        super(FullyConnectedNN, self).__init__()
        
        # Create ModuleList with all the layers
        self.layer_sizes = [input_dim] + hidden_layers + [output_dim]
        self.layers = nn.ModuleList()
        self.activation = activation
        self.loss = nn.MSELoss()
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        
        # Create hidden layers
        for i in range(len(self.layer_sizes)-1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
            if i < (len(self.layer_sizes)-2): # No activation function after the last layer
                self.layers.append(nn.BatchNorm1d(self.layer_sizes[i+1]))
                self.layers.append(activation)
                #self.layers.append(nn.Dropout(p=self.dropout_prob))
                
    def forward(self, input_data):
        """
        We do forward pass on the MSA Transformer embeddings with regression head.
        
        Args: 
            input_data (torch.Tensor): Input data. 
            
        Returns:
            output (torch.Tensor): Output after last layer.
        """
        output = input_data
        for layer in self.layers:
            output = layer(output)
        
        return output
    
def train_epoch(fcn, device, dataloader, optimizer):
    """
    Trains the FCN for one epoch.
    
    Args:
        fcn (nn.Module): FCN we are training. 
        device ("cpu" or "cuda"): Device at which we are doing training. 
        dataloader (torch.DataLoader): Dataset data-loader.
        optimizer (torch.optim): Optimizer for loss-minimization. 
    
    Returns:
        train_loss (float): Train loss over all batches.
    """
    
    train_loss = 0.0
    
    for batch, dists_batch in dataloader:
        # Set model into train mode 
        batch = batch.to(device)
        dists_batch = dists_batch.float().to(device)
        
        # Reset gradients of all tracked variables
        optimizer.zero_grad()
        prediction = fcn(batch).squeeze(-1)
        loss = fcn.loss(prediction, dists_batch)
        
        # Calculate the loss for model outputs
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
            
    return train_loss / len(dataloader)
    
def evaluate(fcn, device, dataloader):
    """
    Evaluate the model.
    
    Args:
        fcn (nn.Module): FCN we are training.
        device ("cpu" or "cuda"): Device at which we are perofmrin evaluation.
        dataloader (torch.DataLoader): Dataset data-loader.

    Returns:
        valid_loss (float): Evaluation loss over all batches.
    """
    eval_loss = 0.0
    fcn.eval()
    store_predictions = []
    store_ground_truths = []
    
    for batch, dists_batch in dataloader:
        with torch.no_grad():
            
            dists_batch = dists_batch.to(device)
            batch = batch.to(device)
            
            # Make predictions
            predictions = fcn(batch).squeeze(-1)
            loss = fcn.loss(predictions, dists_batch)
            store_predictions.append(predictions)
            store_ground_truths.append(dists_batch)
            eval_loss += loss.item()

    av_eval_loss = eval_loss / len(dataloader)
            
    return av_eval_loss, store_predictions, store_ground_truths
    
        
    
    
            
        
          
        
        