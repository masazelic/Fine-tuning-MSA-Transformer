import os
import string
import torch
import random
import numpy as np
import pathlib
import utils
from Bio import SeqIO
from Bio import Phylo

from torch.utils.data import IterableDataset, DataLoader    

def train_val_test_split(pfam_families, ratio_train_test, ratio_val_train, max_depth, msas_folder, dists_folder, normalize_dists=True):
    """ Load sequences from all families with their respective distances and perfrom split on train and test set. 
    
    Args:
        pfam_families (list): List containing all the pfam_families. 
        ratio_train_test (float): Ratio of data that will be used for training.
        ratio_val_train (float): Ratio of train data that will be used for validation
        max_depth (int): Max depth of the MSA found for the family.
        msas_folder (pathlib.Path): Path to the MSA folder.
        dists_folder (pathlib.Path): Path to the distance folder.
    
    Returns: 
        train_data (dict): Dictionary contraining (msa_sequences, distance_matrix) for each family in the train split.
        test_data (dict): Dictionary containing (msa_sequences, distance_matrix) for each family in the test split.
    """
   
    train_data = {}
    val_data = {}
    test_data = {}
    
    
    for family in pfam_families:
        
        msas_path_family = msas_folder / f"{family}_subtree.fasta"
        dists_path_family = dists_folder / f"{family}_subtree.npy"
        
        # Load MSA and distances
        msa_family = utils.read_msa(msas_path_family, max_depth)
        dists_family = np.load(dists_path_family)

        if normalize_dists:
            dists_family = dists_family.astype(np.float64)
            dists_family /= np.max(dists_family)
        
        # Select sequences that will go in train and test / set the seed for the reproduciblity 
        np.random.seed(42)
        all_indices = np.arange(len(msa_family))
        train_indices = np.random.choice(len(msa_family), size=round(ratio_train_test*len(msa_family)), replace=False)
        val_indices = np.random.choice(train_indices, size=round(ratio_val_train*len(msa_family)), replace=False)

        # Remove train_indices from all indices - we're left with test indices
        test_indices = np.setdiff1d(all_indices, train_indices)

        # Remove val_indices from train indices
        train_indices = np.setdiff1d(train_indices, val_indices)
        
        # Make mask for train, val and test
        split_mask = np.full(len(msa_family), -1, dtype=int)

        # We will encode train=0, val=1, test=2
        split_mask[train_indices] = 0 
        split_mask[val_indices] = 1
        split_mask[test_indices] = 2
        
        # Extract train, val and test       
        dists_train_family = dists_family[split_mask == 0, :][:, split_mask == 0]
        dists_val_family = dists_family[split_mask == 1, :][:, split_mask == 1]
        dists_test_family = dists_family[split_mask == 2, :][:, split_mask == 2]
        
        # Extract train and test sequences
        msa_train_family = [msa_family[i] for i in train_indices]
        msa_val_family = [msa_family[i] for i in val_indices]
        msa_test_family = [msa_family[i] for i in test_indices]
        
        train_data[family] = (msa_train_family, dists_train_family)
        val_data[family] = (msa_val_family, dists_val_family)
        test_data[family] = (msa_test_family, dists_test_family)
        
    return train_data, val_data, test_data

def generate_dataloaders_bmDCA(train_data, val_data, test_data):
    """
    Generates dataloader from respective data dictionaries - bmDCA synthetic data.

    Args:
        train_data (dict): Dictionary containing train sequences and respective distances for each family.
        val_data (dict):  Dictionary containing val sequences and respective distances for each family.
        test_data (dict):  Dictionary containing test sequences and respective distances for each family.

    Returns:
        train_dataloader (torch.DataLoader): DataLoader for train data.
        val_dataloader (torch.DataLoader): DataLoader for val data.
        test_dataloader (torch.DataLoader): DataLoader for test data.
    """
    # Define respecitve Datasets
    train_dataset = CustomDatasetbmDCA(train_data, batch_size=32)
    val_dataset = CustomDatasetbmDCA(val_data, batch_size=32)
    test_dataset = CustomDatasetbmDCA(test_data, batch_size=32)
    
    # Define respective DataLoaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=None)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=None)

    return train_dataloader, val_dataloader, test_dataloader


class CustomDatasetbmDCA(IterableDataset):
    
    def __init__(self, family_data, batch_size):
        """
        Custom dataset for loading the data.
        
        Args:   
            family_data (dict): Containing pair of sequences and distances per each family.
            batch_size (int): Number of sequences to include in each batch.
        
        """
        super().__init__()
        self.family_data = family_data
        self.batch_size = batch_size
        
        # Prepare a flattened index to sample families and sequences efficiently
        self.family_to_indices = {family: list(range(len(data[0]))) for family, data in family_data.items()}
        self.families = list(family_data.keys())
        
    def __iter__(self):
        """ Create iterator for sampling. Every iteration, there will be self.batch_size sequences and pairwise distances (for each pair). """
        
        while self.families:
            
            # We pick a random family
            sampling_family = random.choice(self.families)
            
            # Out of all available indices in that family pick batch_size of them
            # If there is less than batch size, pick what is left, and remove that family from list
            if len(self.family_to_indices[sampling_family]) <= self.batch_size:
                indices = self.family_to_indices[sampling_family]
                self.families.remove(sampling_family)
            
            # If there is sufficient amount of elements select batch size
            else:
                indices = random.sample(self.family_to_indices[sampling_family], self.batch_size)
                # We need to eliminate these indices 
                filtered_indices = [x for x in self.family_to_indices[sampling_family] if x not in indices]
                self.family_to_indices[sampling_family] = filtered_indices

            # Extract sequences that belong to that family and indices
            batch_sequences = [self.family_data[sampling_family][0][idx] for idx in indices]
            
            # Extract distances that correspond to those sequences
            indices_mask = np.zeros(self.family_data[sampling_family][1].shape[0], dtype=bool)
            indices_mask[indices] = True
            
            distances = self.family_data[sampling_family][1][indices_mask, :][:, indices_mask]
            
            # We just need to flatten them and convert to tensor
            n_rows = distances.shape[-1]
            triu_indices_batch = np.triu_indices(n_rows, k=1)
            batch_distances = torch.Tensor(distances[triu_indices_batch])
            
            yield batch_sequences, batch_distances

