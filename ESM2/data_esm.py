import os
import numpy as np
import torch
import utils
import pickle
import pathlib
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import IterableDataset, DataLoader 
from Bio import Phylo
from tqdm import tqdm 
from argparse import ArgumentParser

def load_train_val(esm_folder, subsampling=True, ratio=0.05):
    """ Load pre-defined train-val splits. """
    
    # Load data
    train_dict = read_pickle(esm_folder / "train.pkl")
    val_dict = read_pickle(esm_folder / "val.pkl")

    # If we select subsampling=True - takes 20% of train and val sequences
    if subsampling:
        num_train = round(len(train_dict)*ratio)
        num_val = round(len(val_dict)*ratio)

        # Subsample the dictionaries
        train_keys_sample = random.sample(list(train_dict.keys()), num_train)
        val_keys_sample = random.sample(list(val_dict.keys()), num_val)
    
        # Create subsampled dictionaries
        train_dict = {key: train_dict[key] for key in train_keys_sample}
        val_dict = {key: val_dict[key] for key in val_keys_sample}
    

    train_dataloader, len_train = generate_dataloader_esm(train_dict)
    val_dataloader, len_val = generate_dataloader_esm(val_dict)

    return train_dataloader, len_train, val_dataloader, len_val

def load_test(esm_folder, subsampling=True, ratio=0.05):
    """ Load test data. """

    # Load data
    test_dict = read_pickle(esm_folder / "test.pkl")

    # If we select subsampling=True - takes 20% of train and val sequences
    if subsampling:
        num_test = round(len(test_dict)*ratio)

        # Subsample the dictionaries
        train_keys_sample = random.sample(list(test_dict.keys()), num_test)
    
        # Create subsampled dictionaries
        test_dict = {key: test_dict[key] for key in train_keys_sample}

    test_dataloader, len_test = generate_dataloader_esm(test_dict)

    return test_dataloader, len_test

 

def create_paths(esm_folder):
    """ Creates path for ESM data train/test splitting. """

    train_path = esm_folder / "train"
    test_path = esm_folder / "val"

    # Paths for train and test alignments and trees
    train_path_alignments = train_path / "alignments"
    train_path_trees = train_path / "trees"
    train_path_distances = train_path / "distances"

    test_path_alignments = test_path / "alignments"
    test_path_trees = test_path / "trees"
    test_path_distances = test_path / "distances"

    return train_path_alignments, train_path_trees, train_path_distances, test_path_alignments, test_path_trees, test_path_distances

def train_val_split(msa_path, distances_path, ratio=0.15, max_depth=50):
    """ Train-validation split for ESM generated sequences. """
    
    # Load paths to the sequences
    list_msa_seq = os.listdir(msa_path)
    list_distances = os.listdir(distances_path)

    # We need to sort these sequences as they can be randomly read
    list_msa_seq.sort(key=lambda x: (x.split(' ')[0], int(x.split(' ')[1].split('.')[0])))
    list_distances.sort(key=lambda x: (x.split(' ')[0], int(x.split(' ')[1].split('.')[0])))

    # Split indices in train and validation
    indices = np.arange(len(list_msa_seq))
    train_indices, val_indices = train_test_split(indices, test_size=ratio, random_state=42)

    # Extract train and validation pairs
    train_msa_seq = []
    train_distances = []
    validation_msa_seq = []
    validation_distances = []

    for idx in train_indices:
        train_msa_seq.append(list_msa_seq[idx])
        train_distances.append(list_distances[idx])

    for idx in val_indices:
        validation_msa_seq.append(list_msa_seq[idx])
        validation_distances.append(list_distances[idx])

    return train_msa_seq, train_distances, validation_msa_seq, validation_distances

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

def generate_dataloader_esm(data_dict):
    """
    Generate dataloaders from respective dictionary of seq,dists pairs.

    Args: 
        data_dict (dictionary): Dictionary of seq,dists pairs.

    Returns:
        data_loader, length
    """
    # Define respective Dataset
    dataset = CustomDatasetEsm(data_dict)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None)
    length = len(data_dict)

    return data_loader, length 

def create_dictionary(alignments, distances, alignments_path, distances_path):
    """
    Reading from pickle might save time for new runnings?

    Args:
        pickle_path (path): Path to pickle file.
        alignments (list): List of all msa file names. 
        distances (list): List of all distance file names. 
        distances_path (path): Path to distances. 
        alignments_path (path): Path to the synthetic sequences. 

    """
    data = {}

    for i, (msa_path, dists_path) in enumerate(tqdm(zip(alignments, distances))):

        # Read sequences and distances 
        one_path_msa = alignments_path / f"{msa_path}"
        one_distances_path = distances_path / f"{dists_path}"

        msa_seq = utils.read_msa(one_path_msa, 50)
        dists = np.load(one_distances_path)

        # Add to dictionary
        data[i] = (msa_seq, dists)

    return data

def write_pickle(pickle_path, data):
    """ 
    Write data dictionary to pickle file. 

    Args:
        pickle_path (path): Path to the pickle file.
        data (dictionary): Dictionary where we store seq, dists pairs.
    """
    with open(pickle_path, 'wb') as file_dict:
        pickle.dump(data, file_dict)

def read_pickle(pickle_path):
    """
    Read data dictionary from pickle file.

    Args:
        pickle_path (path): Path to the pickle file.
    """
    with open(pickle_path, 'rb') as file_dict:
        data_dict = pickle.load(file_dict)

    return data_dict

class CustomDatasetEsm(IterableDataset):
    """ Custom Dataset for loading sequences obtained using ESM-2 sampling. """
    def __init__(self, data_dict):
        """
        Custom dataset for loading the data. 
        
        Args:
            alignments (list): List of all msa file names. 
            distances (list): List of all distance file names. 
            distances_path (path): Path to distances. 
            alignments_path (path): Path to the synthetic sequences. 
        """
        super().__init__()
        self.data = data_dict

     
    def __iter__(self):
        """ Create iterator for sampling. Since we know that one example from a family has up to 50 sequences that enough for one batch. """
        
        indexes = list(self.data.keys())

        while indexes:
            
            # Select index of a family which will be generated as a batch
            index = int(np.random.choice(indexes, 1))

            # Based on that index we need to select both sequences and respective distances from dictionary
            msa_sequences, dists = self.data[index]

            # Remove the processed element from the list
            indexes.remove(index)

            yield msa_sequences, dists / np.max(dists)

if __name__ == "__main__":

    # This main is to generate pickle file for dictionaries
    # Parser
    parser = ArgumentParser()

    # Parser esm folder
    parser.add_argument('-esmf', '--esm_folder', action='store', dest='esm_folder', default='/content/drive/MyDrive/data/Synthetic_data=50', help='Folder where data for ESM s stored with assumed folder strucutre.')

    # Parser pickle file 
    parser.add_argument('-pf', '--pickle_file', action='store', dest='pickle_file', default='/content/drive/MyDrive/data/Synthetic_data=50')

    # Get arguments
    args = parser.parse_args()
    esm_folder = pathlib.Path(args.esm_folder)
    pickle_file = pathlib.Path(args.pickle_file)

    # Paths
    pickle_train = pickle_file / "train"
    pickle_test = pickle_file / "val"

    # Splits
    train_path_alignments, _, train_path_distances, test_path_alignments, _, test_path_distances = create_paths(esm_folder)
    train_msa_seq, train_distances, val_msa_seq, val_distances = train_val_split(train_path_alignments, train_path_distances)

    test_msa_seq = os.listdir(test_path_alignments)
    test_distances = os.listdir(test_path_distances)

    test_msa_seq.sort(key=lambda x: (x.split(' ')[0], int(x.split(' ')[1].split('.')[0])))
    test_distances.sort(key=lambda x: (x.split(' ')[0], int(x.split(' ')[1].split('.')[0])))

    # Create dictionary
    print("Creating train dictionary...")
    train_data = create_dictionary(train_msa_seq, train_distances, train_path_alignments, train_path_distances)
    
    print("Writing in pickle...")
    write_pickle(pickle_train / "train.pkl", train_data)

    print("Creating val dictionary...")
    val_data = create_dictionary(val_msa_seq, val_distances, train_path_alignments, train_path_distances)

    print("Writing in pickle...")
    write_pickle(pickle_file / "val.pkl", val_data)

    print("Creating test dictionary...")
    test_data = create_dictionary(test_msa_seq, test_distances, test_path_alignments, test_path_distances)

    print("Writing in pickle...")
    write_pickle(pickle_file / "test.pkl", test_data)



    


    

        