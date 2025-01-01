import pathlib
import tqdm 

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

import numpy as np
from numpy.random import default_rng

import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import KFold
import model_FCN 
from sklearn import metrics

import matplotlib.pyplot as plt

# =================================================================================================
# IMPORTANT CONSTANTS

MSAS_FOLDER = pathlib.Path("/content/drive/MyDrive/data/subsampled_msa")
DISTS_FOLDER = pathlib.Path("./distance_matrix")
ATTNS_FOLDER_RAND = pathlib.Path(f"./col_attentions_random")
ATTNS_FOLDER_SUBT = pathlib.Path(f"/content/drive/MyDrive/data/col_attentions_subtree")

pfam_families = ["PF00004", "PF00005", "PF00041", "PF00072", "PF00076", "PF00096", "PF00153", "PF00271", "PF00397", "PF00512", "PF00595", "PF01535", "PF02518", "PF07679", "PF13354"]

# Setting random seed for fixed performance
SEED = 42
rng = np.random.default_rng(SEED)

# Grid search parameters
layer_structure = [[512, 256, 128, 64, 32]]
learning_rates = [0.001]
k_fold = 5
batch_size = 32
num_epochs = [500] 

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==================================================================================================

def create_train_test_sets_per_family(attns, dists, normalize_dists=False, train_size=0.8, ensure_same_size=False, zero_attention_diagonal=False):
    """ Attentions assumed averaged across column dimensions, i.e. 4D tensors. """
    if zero_attention_diagonal:
        attns[:, :, np.arange(attns.shape[2]), np.arange(attns.shape[2])] = 0
    
    assert attns.shape[2] == attns.shape[3]
    
    if normalize_dists:
        dists = dists.astype(np.float64)
        dists /= np.max(dists)
    
    if ensure_same_size:
        dists = dists[:attns.shape[2], :attns.shape[2]]
    
    assert len(dists) == attns.shape[2]
    depth = len(dists)
    n_layers, n_heads, depth, _ = attns.shape
    
    # Train-test split
    n_train = int(depth * train_size)
    
    # Without replacement choose random 70% of MSA depth and create a mask to select training sequences
    train_idx = rng.choice(depth, size=n_train, replace=False)
    split_mask = np.zeros(depth, dtype=bool)
    split_mask[train_idx] = True
        
    # Extract attentions and distances that correspond to the sequences in the training set
    attns_train, attns_test = attns[:, :, split_mask, :][:, :, :, split_mask], attns[:, :, ~split_mask, :][:, :, :, ~split_mask]
    dists_train, dists_test = dists[split_mask, :][:, split_mask], dists[~split_mask, :][:, ~split_mask]
    
    # Attentions abd distance matrices are symmetric, lower triangular part is redundant
    n_rows_train, n_rows_test = attns_train.shape[-1], attns_test.shape[-1]
    triu_indices_train = np.triu_indices(n_rows_train)
    triu_indices_test = np.triu_indices(n_rows_test)
    
    # Train data
    attns_train = attns_train[..., triu_indices_train[0], triu_indices_train[1]]
    attns_train = attns_train.transpose(2, 0, 1).reshape(-1, n_layers * n_heads)
    dists_train = dists_train[triu_indices_train]
    
    # Test data
    attns_test = attns_test[..., triu_indices_test[0], triu_indices_test[1]]
    attns_test = attns_test.transpose(2, 0, 1).reshape(-1, n_layers * n_heads)
    dists_test = dists_test[triu_indices_test]
    
    return (attns_train, dists_train), (attns_test, dists_test), (n_rows_train, n_rows_test)

def hyperparameter_tuning(attns_train, dists_train, layer_structure, learning_rates, num_epochs, batch_size, k_fold, device):
    """ Hyperparameter tuning using K-Fold cross-validation. """
    
    # Preps for K-Fold cross-validation
    train_dataset = TensorDataset(attns_train, dists_train)
    
    splits = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    
    # For storing best model parameters
    best_layer_structure = []
    best_lr = -1.0
    best_mse = -1.0
    
    # Hyperparameter tuning
    for num_epoch in num_epochs:
        for layers in layer_structure:
            for lr in learning_rates:
                print(f"Currently training on: {num_epoch} epochs, {layers} layer structure and {lr} learning rate.")
                avg_fold_mse = 0.0
                
                # Iterate over folds
                for fold, (train_idx, fold_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
                    
                    # Generate train and fold split from this fold
                    train_samples = SubsetRandomSampler(train_idx)
                    fold_samples = SubsetRandomSampler(fold_idx)

                    # Track for this specfic fold split
                    train_loss_fold = []
                    evaluation_loss_fold = []
                    
                    # Create DataLoaders for these splits
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_samples)
                    fold_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=fold_samples)

                    # Define the model and optimizer - we need to train it again for each fold 
                    input_dim = attns_train.shape[1]
                    fcn = model_FCN.FullyConnectedNN(input_dim=input_dim, hidden_layers=layers).to(device)
                    print(fcn)
                    optimizer = optim.Adam(fcn.parameters(), lr=lr)
                    scheduler = ReduceLROnPlateau(optimizer, 'min', 0.5)
                    early_stopping = model_FCN.EarlyStopping(patience=15, delta=0.00005)
                    i = 0
                
                    # Train model for number of epochs
                    for epoch in range(num_epoch):
                        i += 1
                        avg_train_loss = model_FCN.train_epoch(fcn, device, train_loader, optimizer)
                        avg_eval_loss, _, _ = model_FCN.evaluate(fcn, device, fold_loader)

                        # Add them to the respective lists for plotting
                        train_loss_fold.append(avg_train_loss)
                        evaluation_loss_fold.append(avg_eval_loss)
                        
                        #if epoch == (num_epoch - 1):
                        print(f"Epoch {epoch}/{num_epoch}: Train Loss {avg_train_loss:.4f} // Val Loss {avg_eval_loss:.4f}")

                        scheduler.step(avg_eval_loss)
                        early_stopping(avg_eval_loss, fcn)
                        if early_stopping.early_stop:
                            print("Early stopping")
                            break

                    # Plotting loss - for overfitting
                    plt.figure(figsize=(6, 4))
                    plt.plot(np.arange(i), train_loss_fold, label='train loss')
                    plt.plot(np.arange(i), evaluation_loss_fold, label='val loss')
                    plt.title('Test and validation loss on Fold')
                    plt.xlabel('epoch')
                    plt.ylabel('loss')
                    plt.legend()
                    plt.show()

                    
                    # Evaluate model on the fold
                    eval_loss_fold, _, _ = model_FCN.evaluate(fcn, device, fold_loader)
                    print(f"Eval Loss on the {k_fold}th-fold: {eval_loss_fold:.4f}")
                    
                    avg_fold_mse += eval_loss_fold
                
                # Average loss across all folds
                avg_fold_mse = avg_fold_mse / k_fold
                print(f"Average on K-Folds: {avg_fold_mse}")

                
                # Update the parameters of the best model
                if avg_fold_mse > best_mse:
                    
                    best_mse = avg_fold_mse
                    best_layer_structure = layers
                    best_lr = lr
    
    print("Best model hyperparameters:")
    print("Best classifier achitecture: ", best_layer_structure)
    print("Best learning rate: ", best_lr)

def merge_all_families(normalize_dists=True, ensure_same_size=False, zero_attention_diagonal=False):
    """ For loading all the families. """

    attns_train_random = []
    attns_test_random = []
    dists_train_random = []
    dists_test_random = []

    for pfam_family in pfam_families:
        
        # Load path to the specific column attentions and distances
        dists_random = np.load(DISTS_FOLDER / f"{pfam_family}_random.npy")
        attns_random = np.load(ATTNS_FOLDER_RAND / f"{pfam_family}_random_mean_on_cols_symm.npy")

        # Load data
        ((attns_train_r, dists_train_r), (attns_test_r, dists_test_r), (n_rows_train_r, n_rows_test_r)) = create_train_test_sets_per_family(attns_random, dists_random, normalize_dists=normalize_dists, ensure_same_size=ensure_same_size, zero_attention_diagonal=zero_attention_diagonal)
        
        # Convert to torch tensor
        attns_train_r = torch.tensor(attns_train_r)
        attns_test_r = torch.tensor(attns_test_r)
        dists_train_r = torch.tensor(dists_train_r)
        dists_test_r = torch.tensor(dists_test_r)

        # Append data to the respective list
        attns_train_random.append(attns_train_r)
        attns_test_random.append(attns_test_r)
        dists_train_random.append(dists_train_r)
        dists_test_random.append(dists_test_r)

    # Concatenate
    attns_train = torch.cat(attns_train_random, axis=0)
    attns_test = torch.cat(attns_test_random, axis=0)
    dists_train = torch.cat(dists_train_random, axis=0)
    dists_test = torch.cat(dists_test_random, axis=0)

    return attns_train, dists_train, attns_test, dists_test
               
def perform_NN(normalize_dists=True, ensure_same_size=False, zero_attention_diagonal=False):
    """ Perform NN training and evaluation. """
    
    # Dictionary for storing the results
    NN_results = {}

    # Load data after merging from all different families
    attns_train, dists_train, attns_test, dists_test = merge_all_families()

    # I want to plot distribution of the distances - maybe I am unable to lower it down because there is not a signiificant amount 
    # Of sequences that are close by 
    plt.figure(figsize=(6,4))
    plt.hist(dists_train, bins=50)
    plt.title('Histogram of the distances in the training set of this family')
    plt.xlabel('Distances')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(6,4))
    plt.hist(dists_test, bins=50)
    plt.title('Histogram of the distances in the test set of this family')
    plt.xlabel('Distances')
    plt.ylabel('Count')
    plt.show()

    # Tune hyperparameters
    hyperparameter_tuning(attns_train, dists_train, layer_structure, learning_rates, num_epochs, batch_size, k_fold, device)

def train_final_model():
    """ After cross-validation with hyperparameter tuning we are training the model. """

    # Preps for K-Fold cross-validation
    attns_train, dists_train, attns_test, dists_test = merge_all_families()
    train_dataset = TensorDataset(attns_train, dists_train)
    
    splits = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    
    # Parameters from above
    architecture = [512, 256, 128, 64, 32]
    num_epochs = 300
    learning_rate = 0.001
    batch_size = 32

    # Iterate over folds
    for fold, (train_idx, fold_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
        
        # Generate train and fold split from this fold
        train_samples = SubsetRandomSampler(train_idx)
        fold_samples = SubsetRandomSampler(fold_idx)

        # Track for this specfic fold split
        train_loss_fold = []
        evaluation_loss_fold = []
        
        # Create DataLoaders for these splits
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_samples)
        fold_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=fold_samples)

        # Define the model and optimizer - we need to train it again for each fold 
        input_dim = attns_train.shape[1]
        fcn = model_FCN.FullyConnectedNN(input_dim=input_dim, hidden_layers=architecture).to(device)
        print(fcn)
        optimizer = optim.Adam(fcn.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', 0.5)
        early_stopping = model_FCN.EarlyStopping(patience=15, delta=0.00005)
        i = 0
    
        # Train model for number of epochs
        for epoch in range(num_epochs):
            i += 1
            avg_train_loss = model_FCN.train_epoch(fcn, device, train_loader, optimizer)
            avg_eval_loss, _, _ = model_FCN.evaluate(fcn, device, fold_loader)

            # Add them to the respective lists for plotting
            train_loss_fold.append(avg_train_loss)
            evaluation_loss_fold.append(avg_eval_loss)
            
            #if epoch == (num_epoch - 1):
            print(f"Epoch {epoch}/{num_epochs}: Train Loss {avg_train_loss:.4f} // Val Loss {avg_eval_loss:.4f}")

            scheduler.step(avg_eval_loss)
            early_stopping(avg_eval_loss, fcn)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        break

    # Plotting loss - for overfitting
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(i), train_loss_fold, label='train loss')
    plt.plot(np.arange(i), evaluation_loss_fold, label='val loss')
    plt.title('Test and validation loss on Fold')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.save_fig('overfitting.png')

    # Save the model
    path = 'trained_model.pth'
    torch.save(fcn.state_dict(), path)

def evaluate_model(path, device='cpu'):
    """ After we trained the model with best hyperparameters, we are going to evaluate it. """

    # Do data loading 
    _, _, attns_test, dists_test = merge_all_families()
    test_dataset = TensorDataset(attns_test, dists_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load the model
    architecture = [512, 256, 128, 64, 32]
    input_dim = attns_test.shape[1]
    fcn = model_FCN.FullyConnectedNN(input_dim=input_dim, hidden_layers=architecture).to(device)
    fcn.load_state_dict(torch.load(path))

    # Test model
    avg_eval_loss, predictions, ground_truths = model_FCN.evaluate(fcn, device, test_loader)

    # Print the loss on the test data
    print(f"Loss on the test data is: {avg_eval_loss:.4f}")

    # Print the Rˆ2 value on the test data
    predictions_tens = torch.cat(predictions, axis=0)
    ground_truths_tens = torch.cat(ground_truths, axis=0)
    r_squared = metrics.r2_score(ground_truths_tens, predictions_tens)

    print(f"R2-score captured in the test data: {r_squared:.2f}")

    # Do some plots - scatter plot of true values compared to the predictions
    plt.figure(figsize=(6,4))
    plt.plot(ground_truths_tens, predictions_tens)
    plt.title('Dependancy of ground truths and predictions - expecting linear curve')
    plt.xlabel('Ground Truths')
    plt.ylabel('Predictions')
    plt.show()
    plt.savefig('comparison.png')
        
if __name__ == "__main__":
    
    #perform_NN()
    train_final_model()
    evaluate_model('./trained_model.pth')

    

    
    
    
    
    
    
    