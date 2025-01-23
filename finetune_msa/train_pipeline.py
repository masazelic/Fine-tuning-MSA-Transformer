import model_finetune
import model_FCN
import data_esm
import data_bmdca
import utils 


from argparse import ArgumentParser

import pathlib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import peft
import random
from tqdm import tqdm
from sklearn import metrics

# Defining some constants
max_iters = 20
batch_size = 32
learning_rate = 0.0001
r = 16

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

def train_epoch(model, device, dataloader, len_train, optimizer, criterion):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): LoRA model we are fine-tuning.
        device ("cpu" or "cuda"): Device on which we are doing training depending on availability.
        dataloader (torch.Dataloader): Dataset data-loader.
        len_train (int): Number of batches in train dataset.
        optimizer (torch.optim): Optimizer for loss-minimization.
        criterion (nn.Loss): Criterion we are minimizing.
        
    Returns:
        train_loss (float): Train loss averaged for all batches. 
    """
    
    train_loss = 0.0
    
    # Because it is IterableDataset there is no lenght - we need to count it
    num_batches = 0
    
    # Iterate over all batches
    for batch in tqdm(dataloader, total=len_train):
        
        # Tokenize and prepare
        batch_seq, batch_dists = batch
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

def evaluate_epoch(model, device, dataloader, len_val, criterion):
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
    model.eval()
    store_predictions = []
    store_ground_truths = []

    # Iterate over all batches
    for batch in tqdm(dataloader, total=len_val):
        
        # Tokenize and prepare
        batch_seq, batch_dists = batch
        _, _, msa_batch_tokens = model.msa_batch_converter(batch_seq)
        
        # There is no backprop in evaluation
        with torch.no_grad():
            msa_batch_tokens = msa_batch_tokens.to(device)
            batch_dists = batch_dists.to(device)
            
            # Make predictions
            predictions = model(msa_batch_tokens, len(batch_seq)).squeeze(-1)
            loss = criterion(predictions, batch_dists)
            val_loss += loss.detach().float()

            store_predictions.append(predictions)
            store_ground_truths.append(batch_dists)
            
        num_batches += 1
    
    return val_loss / num_batches, store_predictions, store_ground_truths

def train_model(pfam_families, ratio_train_test, ratio_val_train, max_iters, max_depth, r, msas_folder, dists_folder, checkpoint_folder, approach, esm_folder):
    """ Function that does model training for the case of synthetic sequences generated with bmDCA. """
    
    # Save train and validation loss
    train_loss = []
    val_loss = []
    i = 0

    # Checkpoint path - define with the respect to the approach
    checkpoint_folder = checkpoint_folder / f"{approach}_model.pth"

    # Define the data - train, val, test splits 
    if approach == 'bmDCA':
        train_data, len_train, val_data, len_val, test_data, len_test = data_bmdca.train_val_test_split(pfam_families, ratio_train_test, ratio_val_train, max_depth, msas_folder, dists_folder)

    # Define Model, Model LoRA, optimizer, loss function
    model = model_finetune.FineTuneMSATransformer().to(device)
    store_target_modules, store_modules_to_save = utils.get_target_save_modules(model)
    
    config = peft.LoraConfig(r=r, target_modules=store_target_modules, modules_to_save=store_modules_to_save)
    peft_model = peft.get_peft_model(model, config)
    
    optimizer = torch.optim.Adam(peft_model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', 0.9)
    early_stopping = model_FCN.EarlyStopping(patience=5, delta=0.0001)
    criterion = nn.MSELoss()
    
    # Train pipeline
    for epoch in range(max_iters):

        # In every iteration we need to load dataloader - because it is iterable
        i += 1
        if approach == 'bmDCA':
            train_dataloader, val_dataloader, test_dataloader = data_bmdca.generate_dataloaders_bmDCA(train_data, val_data, test_data)
        else:
            train_dataloader, len_train, val_dataloader, len_val = data_esm.load_train_val(esm_folder)
        
        # Set model to train mode
        peft_model.train()
        
        # Train
        avg_train_loss = train_epoch(peft_model, device, train_dataloader, len_train, optimizer, criterion)
        train_loss.append(avg_train_loss.cpu().numpy())
        
        # Evaluate the model on the validation subset
        peft_model.eval()
        avg_eval_loss, _, _ = evaluate_epoch(peft_model, device, val_dataloader, len_val, criterion)
        val_loss.append(avg_eval_loss.cpu().numpy())
        print(f"Epoch {epoch}/{max_iters}: Train {avg_train_loss:.4f} // Val {avg_eval_loss:.4f}")

        # Scheduler step and early stopping
        scheduler.step(avg_eval_loss)
        early_stopping(avg_eval_loss, peft_model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Plot loss curves
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(i), train_loss, label='train loss')
    plt.plot(np.arange(i), val_loss, label='val loss')
    plt.title('Train and validation loss on Fold')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('overfitting.png')

    # Define Final Model, Model LoRA, optimizer, loss function
    final_model = model_finetune.FineTuneMSATransformer().to(device)
    store_target_modules, store_modules_to_save = utils.get_target_save_modules(final_model)
    
    config = peft.LoraConfig(r=r, target_modules=store_target_modules, modules_to_save=store_modules_to_save)
    final_peft_model = peft.get_peft_model(final_model, config)

    final_peft_model.load_state_dict(early_stopping.best_model_state)

    # Evaluate final model
    if approach == 'bmDCA':
        _, val_dataloader, _ = data_bmdca.generate_dataloaders_bmDCA(train_data, val_data, test_data)
    else:
        _, _, val_dataloader, len_val = data_esm.load_train_val(esm_folder)

    avg_eval_loss_fin, predictions, ground_truths = evaluate_epoch(final_peft_model, device, val_dataloader, len_val, criterion)
    print(f"Final validation loss: {avg_eval_loss_fin:.4f}")

    # Print the Rˆ2 value on the val data
    predictions_tens = torch.cat(predictions, axis=0).detach().cpu().numpy()
    ground_truths_tens = torch.cat(ground_truths, axis=0).detach().cpu().numpy()
    r_squared = metrics.r2_score(ground_truths_tens, predictions_tens)

    print(f"R2-score captured in the test data: {r_squared:.2f}")

    # Save model
    torch.save(early_stopping.best_model_state, checkpoint_folder)

def test_model(pfam_families, ratio_train_test, ratio_val_train, max_iters, max_depth, r, msas_folder, dists_folder, checkpoint_folder, approach, esm_folder):
    """ Function that does model training for the case of synthetic sequences generated with bmDCA. """

    # Checkpoint path - define with the respect to the approach
    checkpoint_folder = checkpoint_folder / f"{approach}_model.pth"

    # Define the data - train, val, test splits 
    if approach == 'bmDCA':
        train_data, _, val_data, _, test_data, len_test = data_bmdca.train_val_test_split(pfam_families, ratio_train_test, ratio_val_train, max_depth, msas_folder, dists_folder)
    

    # Define Model, Model LoRA, optimizer, loss function
    model = model_finetune.FineTuneMSATransformer().to(device)
    store_target_modules, store_modules_to_save = utils.get_target_save_modules(model)
    
    config = peft.LoraConfig(r=r, target_modules=store_target_modules, modules_to_save=store_modules_to_save)
    peft_model = peft.get_peft_model(model, config)
    peft_model.load_state_dict(torch.load(checkpoint_folder))
    
    total_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    criterion = nn.MSELoss()

    # Test model
    if approach == 'bmDCA':
        _, _, test_dataloader = data_bmdca.generate_dataloaders_bmDCA(train_data, val_data, test_data)
    else:
        test_dataloader, len_test = data_esm.load_test(esm_folder)

    avg_eval_loss_fin, predictions, ground_truths = evaluate_epoch(peft_model, device, test_dataloader, len_test, criterion)
    print(f"Final validation loss: {avg_eval_loss_fin:.4f}")

    # Print the Rˆ2 value on the test data
    predictions_tens = torch.cat(predictions, axis=0).detach().cpu().numpy()
    ground_truths_tens = torch.cat(ground_truths, axis=0).detach().cpu().numpy()
    r_squared = metrics.r2_score(ground_truths_tens, predictions_tens)

    print(f"R2-score captured in the test data: {r_squared:.2f}")

    # Plot
    plt.figure(figsize=(6,4))
    plt.scatter(ground_truths_tens, predictions_tens, s=5)
    plt.title('Dependancy of ground truths and predictions - expecting linear curve')
    plt.xlabel('Ground Truths')
    plt.ylabel('Predictions')
    plt.show()
    plt.savefig('comparison.png')

 
if __name__ == "__main__":
    
    # Parsing command-line options
    parser = ArgumentParser()
    
    # Ratio train-test
    parser.add_argument('-tt', '--ratio_train_test', action='store', dest='ratio_train_test', default='0.8', help='Train data ratio.')
    
    # Ratio train-val
    parser.add_argument('-vt', '--ratio_val_train', action='store', dest='ratio_val_train', default='0.1', help='Validation data ratio.')
    
    # Max depth
    parser.add_argument('-md', '--max_depth', action='store', dest='max_depth', default='600', help='Max depth MSA sequences from each family.')
    
    # MSAs folder for bmDCA
    parser.add_argument('-mf', '--msas_folder', action='store', dest='msas_folder', default='./subsampled_msa', help='MSAs folder path.')
    
    # Dists folder for bmDCA
    parser.add_argument('-df', '--dists_folder', action='store', dest='dists_folder', default='./distance_matrix', help='Distance matrices folder path.')

    # Checkpoint folder
    parser.add_argument('-cp', '--checkpoint_folder', action='store', dest='checkpoint_folder', default='/content/drive/MyDrive/data/checkpoints', help='Folder that stores model checkpoints over training.')

    # Approach
    parser.add_argument('-a', '--approach', action='store', dest='approach', default='esm', help='Tells which synthetic sequences to use. Can be ESM or bmDCA.')

    # Folder where data for ESM is stored - assumes following folder structure
    # - Folder with data (argument)
    #   - train
    #     - alignments
    #     - trees
    #   - val
    #     - alignments
    #     - trees
    parser.add_argument('-esmf', '--esm_folder', action='store', dest='esm_folder', default='/content/drive/MyDrive/data/Synthetic_data=50', help='Folder where data for ESM s stored with assumed folder strucutre.')
    
    # Get arguments
    args = parser.parse_args()
    ratio_train_test = float(args.ratio_train_test)
    ratio_val_train = float(args.ratio_val_train)
    max_depth = int(args.max_depth)
    msas_folder = pathlib.Path(args.msas_folder)
    dists_folder = pathlib.Path(args.dists_folder)
    checkpoint_folder = pathlib.Path(args.checkpoint_folder)
    approach = args.approach
    esm_folder = pathlib.Path(args.esm_folder)

    if approach == "bmDCA":
        train_model(pfam_families, ratio_train_test, ratio_val_train, max_iters, max_depth, r, msas_folder, dists_folder, checkpoint_folder, approach)
        test_model(pfam_families, ratio_train_test, ratio_val_train, max_iters, max_depth, r, msas_folder, dists_folder, checkpoint_folder, approach)
    
    if approach == "esm":
       #train_model(pfam_families, ratio_train_test, ratio_val_train, max_iters, max_depth, r, msas_folder, dists_folder, checkpoint_folder, approach, esm_folder)
        test_model(pfam_families, ratio_train_test, ratio_val_train, max_iters, max_depth, r, msas_folder, dists_folder, checkpoint_folder, approach, esm_folder)

    



    
    
    
    
    
        
        
    
    
    
    
    
    