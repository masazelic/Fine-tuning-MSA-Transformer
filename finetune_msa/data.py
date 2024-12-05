import os
import string
import torch
import random
import numpy as np
import pathlib
import itertools
from Bio import SeqIO
from Bio import Phylo
from sklearn.model_selection import train_test_split

from torch.utils.data import IterableDataset, DataLoader    

def remove_insertions(sequence):
    """ Removes any insertions into the sequences. Needed to load aligned sequences in an MSA."""
    
    # Making dictionary where each lowercase ascii letter is key and value is set to None
    deletekeys = dict.fromkeys(string.ascii_lowercase) 
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    
    return sequence.translate(translation)

def read_msa(filename, nseq):
    """ Reads the first nseq sequences from an MSA file in fasta format, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def train_val_test_split(pfam_families, ratio_train_test, ratio_val_train, max_depth, msas_folder, dists_folder):
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
        msa_family = read_msa(msas_path_family, max_depth)
        dists_family = np.load(dists_path_family)
        
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

def train_val_split(msa_path, tree_path, ratio=0.15, max_depth=50):
    """ Train-validation split for ESM generated sequences. """
    
    # Load paths to the sequences
    list_msa_seq = os.listdir(msa_path)
    list_trees = os.listdir(tree_path)

    # We need to sort these sequences as they can be randomly read
    list_msa_seq.sort(key=lambda x: (x.split(' ')[0], int(x.split(' ')[1].split('.')[0])))
    list_trees.sort(key=lambda x: (x.split(' ')[0], int(x.split(' ')[1].split('.')[0])))

    # Split indices in train and validation
    indices = np.arange(len(list_msa_seq))
    train_indices, val_indices = train_test_split(indices, test_size=ratio, random_state=42)

    # Extract train and validation pairs
    train_msa_seq = []
    train_trees = []
    validation_msa_seq = []
    validation_trees = []

    for idx in train_indices:
        train_msa_seq.append(list_msa_seq[idx])
        train_trees.append(list_trees[idx])

    for idx in val_indices:
        validation_msa_seq.append(list_msa_seq[idx])
        validation_trees.append(list_trees[idx])

    return train_msa_seq, train_trees, validation_msa_seq, validation_trees

def create_paths(esm_folder):
    """ Creates path for ESM data train/test splitting. """

    train_path = esm_folder / "train"
    test_path = esm_folder / "val"

    # Paths for train and test alignments and trees
    train_path_alignments = train_path / "alignments"
    train_path_trees = train_path / "trees"

    test_path_alignments = test_path / "alignments"
    test_path_trees = test_path / "trees"

    return train_path_alignments, train_path_trees, test_path_alignments, test_path_trees

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

def generate_dataloaders_esm(train_msa_seq, train_trees, train_path_alignments, train_path_trees, val_msa_seq, val_trees, test_path_alignments, test_path_trees):
    """
    Generate dataloaders from respective sequences and trees.

    Args: 
        train_msa_seq (list): Names of MSA alignment files that belong to train subset.
        train_trees (list): Names of tree files that belong to train subset.
        val_msa_seq (list): Names of MSA alignment files that belong to val subset.
        val_trees (list): Names of tree files that belong to val subset.
        test_path_alignments (path): Path to the test aligments. 
        test_path_trees (path): Path to the test trees.

    Returns:
    """
    # Define respective Datasets for train and val
    train_dataset = CustomDatasetEsm(train_msa_seq, train_trees, train_path_alignments, train_path_trees)
    val_dataset = CustomDatasetEsm(val_msa_seq, val_trees, train_path_alignments, train_path_trees)

    test_msa_seq = os.listdir(test_path_alignments)
    test_trees = os.listdir(test_path_trees)

    # Sort test dataset
    test_msa_seq.sort(key=lambda x: (x.split(' ')[0], int(x.split(' ')[1].split('.')[0])))
    test_trees.sort(key=lambda x: (x.split(' ')[0], int(x.split(' ')[1].split('.')[0])))
    test_dataset = CustomDatasetEsm(test_msa_seq, test_trees, test_path_alignments, test_path_trees)

    # Define respective DataLoaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=None)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=None)

    return train_dataloader, val_dataloader, test_dataloader

def create_distance_matrix(msa_sequences, tree):
    """ Provided MSA sequences (their names) and tree create distance matrix. """
    num_sequences = len(msa_sequences)

    # Create distance matrix that will store these 
    distance_matrix = np.zeros((num_sequences, num_sequences))

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            # Calculate distance between sequence pairs using tree
            distance = tree.distance(msa_sequences[i][0], msa_sequences[j][0])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix
    
    return distance_matrix
        
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

class CustomDatasetEsm(IterableDataset):
    """ Custom Dataset for loading sequences obtained using ESM-2 sampling. """
    def __init__(self, alignments, trees, alignments_path, trees_path):
        """
        Custom dataset for loading the data. 
        
        Args:   
            tree_path (string): Path to the trees along which synthetic sequences are generated. 
            alignments_path (string): Path to the synthetic sequences. 
        """
        super().__init__()

        # Define the path to trees and alignments
        self.alignments_path = alignments_path
        self.trees_path = trees_path

        # Esentially os.listdir of these paths
        self.list_tree_path = trees
        self.list_alignments_path = alignments
        
        self.max_depth = 50
        
    def __iter__(self):
        """ Create iterator for sampling. Since we know that one example from a family has up to 50 sequences that enough for one batch. """
        
        while self.list_tree_path:
            
            # Select index of a family which will be generated as a batch
            index = int(np.random.choice(len(self.list_tree_path), 1))
            # Based on that index we need to select both sequences and respective tree
            tree_path = self.trees_path / f"{self.list_tree_path[index]}"
            msa_path = self.alignments_path / f"{self.list_alignments_path[index]}"
           
            msa_sequences = read_msa(msa_path, self.max_depth)
            tree = Phylo.read(tree_path, "newick")

            # Create their respective distances
            distance_matrix = create_distance_matrix(msa_sequences, tree)
            n_rows = distance_matrix.shape[-1]
            triu_indices_batch = np.triu_indices(n_rows, k=1)
            batch_distances = distance_matrix[triu_indices_batch] 

            # Remove the processed element from the lists
            self.list_tree_path.pop(index)
            self.list_alignments_path.pop(index)

            yield msa_sequences, batch_distances


