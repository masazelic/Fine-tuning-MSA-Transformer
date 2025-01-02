import torch
import torch.nn as nn
import esm 

class FineTuneMSATransformer(nn.Module):
    """
    Fine-tuning MSA Transformer to infer patristic distance.
    
    Args:
        input_dim (int): Input dimension.
        architecture (list): List containing the architecture of FCN that comes on top of this model to infer distancies.
        output_dim (int): Since we do regression to predict patristic distance output_dim = 1.
        dropout_p (float): Dropout probability for regularization.
        activation (callable): The activation function to apply after each hidden layer.
    """
    def __init__(self, input_dim=144, architecture=[256,128,64], output_dim=1, dropout=0.0):
        
        super().__init__()
        
        # Create a model that is pretrained MSA Transformer that is going to be fine-tuned with LoRA
        self.msa_transformer, self.msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_batch_converter = self.msa_alphabet.get_batch_converter()
    
        # Define some parameters
        self.n_heads = 12
        self.n_layers = 12
        
        # FCN model that is going to be build on top of MSA Transformer
        self.layer_sizes = [input_dim] + architecture + [output_dim]
        self.layers = nn.ModuleDict()
        
        for i in range(len(self.layer_sizes) - 1):
            
            self.layers[f'finetune_linear_{i}'] = nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
            
            # No activation or dropout function after last layer
            if i < (len(self.layer_sizes) - 2):
                self.layers[f'finetune_activation_{i}'] = nn.ReLU()
                self.layers[f'finetune_dropout_{i}'] = nn.Dropout(dropout_p)
            
    def extract_collumn_attentions(self, results, batch_size):
        """
        Extracts information from column attentions for downstream model.

        Args:
            results (dict): Result of running MSA Transformer.
            batch_size (int): Batch size (can be different than fixed value, depending on how many sequences is left in the family).
            
        Returns:
            upper_triangle_embeddings (torch.Tensor): Shape (#sequence pairs in batch, n_heads*n_layers).
        """
        # Extract column attentions
        attns_mean_on_colls_symm = results['col_attentions'][0, ...].mean(axis=2)
        attns_mean_on_colls_symm += attns_mean_on_colls_symm.permute(0,1,3,2).clone()
        
        # We need to extract data properly for downstream, i.e. we need to have 144 dim embeddings 
        # For each pairwise sequence setup
        indices = torch.triu_indices(batch_size, batch_size, offset=1)
        upper_triangle = attns_mean_on_colls_symm[..., indices[0], indices[1]]
        upper_triangle_embeddings = upper_triangle.permute(2,0,1).reshape(-1, self.n_layers * self.n_heads)
        
        return upper_triangle_embeddings
    
    def forward(self, msa_batch_tokens, batch_size): 
        """
        Perform forward pass. 
        
        Args:
            input_data (list): List containing batch_size of protein sequences aligned (MSA).
            batch_size (int): Batch size (can be different than fixed value, depending on how many sequences is left in the family).
        
        Returns:
            output (torch.Tensor): Predicted patristic distance for sequences in batch.
        """
        
        # Run MSA transformer - we never set torch.no_grad or eval mode
        results = self.msa_transformer(msa_batch_tokens, repr_layers=[12], need_head_weights=True)
        
        # Run function to get whats needed for predicting distances
        upper_triangle_embeddings = self.extract_collumn_attentions(results, batch_size)

        # Run dowstream FCN
        output = upper_triangle_embeddings
        for i in range(len(self.layer_sizes) - 1):
            
            # Apply linear layer
            output = self.layers[f'finetune_linear_{i}'](output)
            # Skip activation and dropout after the last layer
            if i < len(self.layer_sizes) - 2:
                output = self.layers[f'finetune_activation_{i}'](output)
                output = self.layers[f'finetune_dropout_{i}'](output)
        
        return output
    