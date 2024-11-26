import model_finetune
import data
import utils 


from argparse import ArgumentParser

import pathlib
import torch
import torch.nn as nn
import peft 

# Defining some constants
max_iters = 500
batch_size = 32
learning_rate = 0.001

pfam_families = [
    "PF00004",
    "PF00005",
    "PF00041",
    "PF00072",
    "PF00076",
    "PF00096",
    "PF00153",
    "PF00271",
    "PF00397",
    "PF00512",
    "PF00595",
    "PF01535",
    "PF02518",
    "PF07679",
    "PF13354"
]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, device, dataloader, optimizer, criterion):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): LoRA model we are fine-tuning.
        device ("cpu" or "cuda"): Device on which we are doing training depending on availability.
        dataloader (torch.Dataloader): Dataset data-loader.
        optimizer (torch.optim): Optimizer for loss-minimization.
        criterion (nn.Loss): Criterion we are minimizing.
        
    Returns:
        train_loss (float): Train loss averaged for all batches. 
    """
    
    train_loss = 0.0
    
    # Because it is IterableDataset there is no lenght - we need to count it
    num_batches = 0
    
    # Iterate over all batches
    for batch_seq, batch_dists in dataloader:
        
        # Tokenize and prepare
        _, _, msa_batch_tokens = model.msa_batch_converter(batch_seq)
        msa_batch_tokens = msa_batch_tokens.to(device)
        batch_dists = batch_dists.float().to(device)
        
        # Reset gradients of all tracked variables
        optimizer.zero_grad()
        prediction = model(msa_batch_tokens, len(batch_seq)).squeeze(-1)
        loss = criterion(prediction, batch_dists)
        train_loss += loss.detach().float()
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        num_batches += 1
    
    return train_loss / num_batches

def evaluate_epoch(model, device, dataloader, criterion):
    """
    Evaluate model on one epoch.
    
    Args:
        model (nn.Module): LoRA model we are fine-tuning (set to evaluate).
        device ("cpu" or "cuda"): Device on which we are doing evaluation depending on availability.
        dataloader (torch.Dataloader): Dataset data-loader.
        criterion (nn.Loss): Criterion we are minimizing.
        
    Return:
        val_loss (float): Validation loss averaged for all batches.
    """
    
    val_loss = 0.0
    num_batches = 0
    
    # Iterate over all batches
    for batch_seq, batch_dists in dataloader:
        
        # Tokenize and prepare
        _, _, msa_batch_tokens = model.msa_batch_converter(batch_seq)
        
        # There is no backprop in evaluation
        with torch.no_grad():
            msa_batch_tokens = msa_batch_tokens.to(device)
            batch_dists = batch_dists.to(device)
            
            # Make predictions
            predictions = model(msa_batch_tokens, len(batch_seq)).squeeze(-1)
            loss = criterion(predictions, batch_dists)
            val_loss += loss.detach().float()
            
        num_batches += 1
    
    return val_loss / num_batches
            

if __name__ == "__main__":
    
    # Parsing command-line options
    parser = ArgumentParser()
    
    # Ratio train-test
    parser.add_argument('-tt', '--ratio_train_test', action="store", dest="ratio_train_test", default='0.8', help='Train data ratio.')
    
    # Ratio train-val
    parser.add_argument('-vt', '--ratio_val_train', action="store", dest="ratio_val_train", default='0.1', help='Validation data ratio.')
    
    # Max depth
    parser.add_argument('-md', '--max_depth', action='store', dest='max_depth', default='600', help='Max depth MSA sequences from each family.')
    
    # MSAs folder 
    parser.add_argument('-mf', '--msas_folder', action='store', dest='msas_folder', default='/content/drive/MyDrive/data/subsampled_msa', help='MSAs folder path.')
    
    # Dists folder
    parser.add_argument('-df', '--dists_folder', action='store', dest='dists_folder', default='/content/drive/MyDrive/data/distance_matrix', help='Distance matrices folder path.')
    
    
    # Get arguments
    args = parser.parse_args()
    ratio_train_test = float(args.ratio_train_test)
    ratio_val_train = float(args.ratio_val_train)
    max_depth = int(args.max_depth)
    msas_folder = pathlib.Path(args.msas_folder)
    dists_folder = pathlib.Path(args.dists_folder)
    
    # Define the data - train, val, test splits 
    train_data, val_data, test_data = data.train_val_test_split(pfam_families, ratio_train_test, ratio_val_train, max_depth, msas_folder, dists_folder)
    
    # Define Model, Model LoRA, optimizer, loss function
    model = model_finetune.FineTuneMSATransformer().to(device)
    store_target_modules, store_modules_to_save = utils.get_target_save_modules(model)
    
    config = peft.LoraConfig(r=8, target_modules=store_target_modules, modules_to_save=store_modules_to_save)
    peft_model = peft.get_peft_model(model, config)
    
    optimizer = torch.optim.Adam(peft_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Train pipeline
    for epoch in range(max_iters):

        # In every iteration we need to load dataloader - because it is iterable
        train_dataloader, val_dataloader, test_dataloader = data.generate_dataloaders(train_data, val_data, test_data)
        
        # Set model to train mode
        peft_model.train()
        
        # Train
        avg_train_loss = train_epoch(peft_model, device, train_dataloader, optimizer, criterion)
        
        # Evaluate the model on the validation subset
        peft_model.eval()
        avg_eval_loss = evaluate_epoch(peft_model, device, val_dataloader, criterion)
        print(f"Epoch {epoch}/{max_iters}: Train {avg_train_loss:.4f} // Val {avg_eval_loss:.4f}")
    
    avg_eval_loss_fin = evaluate_epoch(peft_model, device, val_dataloader, criterion)
    print(f"Final validation loss: {avg_eval_loss_fin:.4f}")
    
    
        
        
    
    
    
    
    
    