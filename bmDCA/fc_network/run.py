import pathlib
import tqdm 
import pickle

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

import numpy as np
from numpy.random import default_rng

import model_FCN 
import load_data

import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import KFold
from sklearn import metrics

import matplotlib.pyplot as plt
from argparse import ArgumentParser

np.random.seed(26)

def train_final_model(pfam_families, dists_folder, attns_folder, train_indexes, best_parameters, device, approach, k_fold=5):
    """ 
    Training final model with cross-validated hyperparameters. 

    Args:
        pfam_family (list): List containing all the family names.
        dists_folder (path.Pathlib): Path to the folder where distances are stored.
        attns_folder (path.Pathlib): Path to the folder where attentions are stored.
        train_indexes (np.array): Indexes that will be part of train set from all families. 
        best_parameters (dictionary): Dictionary containing best parameters obtained by grid search. 
        device: Either cpu or cuda. 
        approach: Either subtree or random.
        k_fold (int): Number of folds.
  
    """

    # Preps for K-Fold cross-validation
    attns_train, dists_train, _, _ = load_data.merge_all_families(pfam_families, dists_folder, attns_folder, train_indexes, approach)
    train_dataset = TensorDataset(attns_train, dists_train)
    
    splits = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    
    # Parameters from above
    architecture = best_parameters['layer_structure']
    num_epochs = best_parameters['num_epochs']
    learning_rate = best_parameters['learning_rate']
    batch_size = best_parameters['batch_size']

    # Iterate over folds
    for _, (train_idx, fold_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
        
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
        early_stopping = model_FCN.EarlyStopping(patience=25, delta=0.00005)
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
        plt.savefig('overfitting4.png')

        # Save the model
        path = 'trained_{approach}.pth'
        torch.save(early_stopping.best_model_state, path)
        
        break

def evaluate_model(pfam_families, dists_folder, attns_folder, train_indexes, approach, path, best_parameters, device='cpu'):
    """ 
    After we trained the model with best hyperparameters, we are going to evaluate it. 
    
    Args:
        pfam_family (list): List containing all the family names.
        dists_folder (path.Pathlib): Path to the folder where distances are stored.
        attns_folder (path.Pathlib): Path to the folder where attentions are stored.
        train_indexes (np.array): Indexes that will be part of train set from all families. 
        approach: Either subtree or random.
        path (str): Path where best model is stored.
        best_parameters (dict): Dictionary where best parameters are stored.
        device: Either cpu or cuda.
    """

    # Do data loading 
    _, _, attns_test, dists_test = load_data.merge_all_families(pfam_families, dists_folder, attns_folder, train_indexes, approach)
    test_dataset = TensorDataset(attns_test, dists_test)
    test_loader = DataLoader(test_dataset, batch_size=best_parameters['batch_size'])

    # Load the model
    architecture = best_parameters['layer_structure']
    input_dim = attns_test.shape[1]
    fcn = model_FCN.FullyConnectedNN(input_dim=input_dim, hidden_layers=architecture).to(device)
    fcn.load_state_dict(torch.load(path))

    # Test model
    avg_eval_loss, predictions, ground_truths = model_FCN.evaluate(fcn, device, test_loader)

    # Print the loss on the test data
    print(f"Loss on the test data is: {avg_eval_loss:.4f}")

    # Print the Rˆ2 value on the test data
    predictions_tens = torch.cat(predictions, axis=0).detach().cpu().numpy()
    ground_truths_tens = torch.cat(ground_truths, axis=0).detach().cpu().numpy()
    r_squared = metrics.r2_score(ground_truths_tens, predictions_tens)

    print(f"R2-score captured in the test data: {r_squared:.2f}")

    # Do some plots - scatter plot of true values compared to the predictions
    plt.figure(figsize=(6,4))
    plt.scatter(ground_truths_tens, predictions_tens, s=5)
    plt.title('Dependancy of ground truths and predictions - expecting linear curve')
    plt.xlabel('Ground Truths')
    plt.ylabel('Predictions')
    plt.show()
    plt.savefig('comparison4.png')
        
if __name__ == "__main__":
    
    pfam_families = ["PF00004", "PF00005", "PF00041", "PF00072", "PF00076", "PF00096", "PF00153", "PF00271", "PF00397", "PF00512", "PF00595", "PF01535", "PF02518", "PF07679", "PF13354"]
    
    # Parsing command-line options
    parser = ArgumentParser()
    
    # Ratio train-test
    parser.add_argument('-tt', '--ratio_train_test', action='store', dest='ratio_train_test', default='0.8', help='Train data ratio.')
    
    # Attns folder
    parser.add_argument('-af', '--attentions_folder', action='store', dest='attentions_folder', default='/content/drive/MyDrive/data/col_attentions_random', help='Folder where attention matrices are stored.')
    
    # Distances folder
    parser.add_argument('-df', '--distances_folder', action='store', dest='distances_folder', default='/content/drive/MyDrive/data/distance_matrix', help='Folder where distance matrices are stored.')
    
    # Approach
    parser.add_argument('-a', '--approach', action='store', dest='approach', default='random', help='Either subtree or random subsampling approach obtained sequences.')
    
    # Train or Test
    parser.add_argument('-r', '-train_test', action='store', dest='train_test', default='train', help='Either train or test model.')
    
    # Path to the best parameters
    parser.add_argument('-bp', '--best_parameters', action='store', dest='best_parameters_path', default='/content/drive/MyDrive/Fine-tuning-MSA-Transformer/Fine-tuning-MSA-Transformer/bmDCA/fc_network/parameters/best_parameters_subtree.pkl',help='Path to the best parameters.')
    
    # Get arguments
    args = parser.parse_args()
    ratio_train_test = float(args.ratio_train_test)
    attns_folder = pathlib.Path(args.attentions_folder)
    dists_folder = pathlib.Path(args.distances_folder)
    approach = args.approach
    train_test = args.train_test
    best_parameters_path = args.best_parameters_path
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load parameters of the model
    with open(best_parameters_path, 'rb') as f:
        best_parameters = pickle.load(f)
        
    # Generate train indexes 
    train_indexes = []
    for pfam_family in pfam_families:
        
        dists_matrix = np.load(dists_folder / f"{pfam_family}_{approach}.npy")
        n_seq = dists_matrix.shape[0]
        
        train_indexes.append(np.random.choice(n_seq, round(n_seq * ratio_train_test), replace=False))
        
    if train_test == 'train':
        train_final_model(pfam_families, dists_folder, attns_folder, train_indexes, best_parameters, device, approach)
    else:
        evaluate_model(pfam_families, dists_folder, attns_folder, train_indexes, approach, './trained_{approach}.pth', best_parameters, device)

    

    
    
    
    
    
    
    