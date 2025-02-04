import load_data
import model_FCN 

import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt

import numpy as np
import pathlib
import pickle

from argparse import ArgumentParser

# Setting random seed for fixed performance
np.random.seed(26)

def hyperparameter_tuning(attns_train, dists_train, layer_structure, learning_rates, num_epochs, batch_size, k_fold, device, approach):
    """ 
    Hyperparameter tuning using K-Fold cross-validation. 
    
    Args:
        attns_train (torch.Tensor): Train embeddings for all families. 
        dists_train (torch.Tensor): Train distances for all families. 
        layer_structure (list): List of possible architectures.
        learning_rates (list): List of possible learning rates.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size.
        k_fold (int): Number of folds for cross-validation. 
        device: Either 'cpu' or 'cuda'.
        approach: Either 'random' or 'subsampling'.
    """
    
    # Preps for K-Fold cross-validation
    train_dataset = TensorDataset(attns_train, dists_train)
    
    splits = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    
    # For storing best model parameters
    best_layer_structure = []
    best_lr = -1.0
    best_mse = -1.0
    
    # Hyperparameter tuning
    
    for layers in layer_structure:
        for lr in learning_rates:
            print(f"Currently training on: {num_epochs} epochs, {layers} layer structure and {lr} learning rate.")
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
    
    # Save to dictionary
    best_parameters = {'layer_structure': best_layer_structure, 'learning_rate': best_lr, 'batch_size': batch_size, 'num_epochs': num_epochs}
    
    with open(f'best_parameters_{approach}.pkl', 'wb') as f:
        pickle.dump(best_parameters, f)
    
if __name__ == "main":
        
    pfam_families = ["PF00004", "PF00005", "PF00041", "PF00072", "PF00076", "PF00096", "PF00153", "PF00271", "PF00397", "PF00512", "PF00595", "PF01535", "PF02518", "PF07679", "PF13354"]
    
    # Parsing command line options 
    parser = ArgumentParser()
    
    # Ratio train-test
    parser.add_argument('-tt', '--ratio_train_test', action='store', dest='ratio_train_test', default='0.8', help='Train data ratio.')
    
    # Attns folder
    parser.add_argument('-af', '--attentions_folder', action='store', dest='attentions_folder', default='/content/drive/MyDrive/data/col_attentions_random', help='Folder where attention matrices are stored.')
    
    # Distances folder
    parser.add_argument('-df', '--distances_folder', action='store', dest='distances_folder', default='/content/drive/MyDrive/data/distance_matrix', help='Folder where distance matrices are stored.')
    
    # Approach
    parser.add_argument('-a', '--approach', action='store', dest='approach', default='subtree', help='Either subtree or random subsampling approach obtained sequences.')
    
    # Get arguments
    args = parser.parse_args()
    ratio_train_test = float(args.ratio_train_test)
    attns_folder = pathlib.Path(args.attentions_folder)
    dists_folder = pathlib.Path(args.distances_folder)
    approach = args.approach
    
    # Important!
    # Parameters for grid search 
    layer_structure = [[512, 256, 128, 64, 32], [256, 128, 64], [64, 32, 16]]
    learning_rates = [0.001, 0.0001, 0.00001]
    k_fold = 5
    batch_size = 32
    num_epochs = 300
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate train indexes 
    train_indexes = []
    for pfam_family in pfam_families:
        
        dists_matrix = np.load(dists_folder / f"{pfam_family}_{approach}.npy")
        n_seq = dists_matrix.shape[0]
        
        train_indexes.append(np.random.choice(n_seq, round(n_seq * ratio_train_test), replace=False))
        
    # Load train data
    attns_train, dists_train, _, _ = load_data.merge_all_families(pfam_families, dists_folder, attns_folder, train_indexes, approach, normalize_dists=True)
    
    # Hyperparameter tuning
    hyperparameter_tuning(attns_train, dists_train, layer_structure, learning_rates, num_epochs, batch_size, k_fold, device, approach)
    