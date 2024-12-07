from argparse import ArgumentParser
import pathlib
import data_esm
import utils
import os
import numpy as np
from Bio import Phylo
from tqdm import tqdm

def create_and_store_dists_matrices(msas_path, trees_path, dists_path, max_depth=50):
    """
    Creates and stores distance matrices so they can be stored externally instead of calculated each time - to save computational time. 
    
    Args:
        msa_path (path): Path to all alignments train files.
        trees_path (path): Path to all train tree files.
        dists_path (path): Path to the folder where we will store all train distances. 
        max_depth (int, default=50): Max depth of the the fasta file, we know it is 50. 
    
    """

    # Load all the files in directories
    list_msa_seq = os.listdir(msas_path)
    list_trees = os.listdir(trees_path)
    
    # We need to sort these sequences as they can be randomly read
    list_msa_seq.sort(key=lambda x: (x.split(' ')[0], int(x.split(' ')[1].split('.')[0])))
    list_trees.sort(key=lambda x: (x.split(' ')[0], int(x.split(' ')[1].split('.')[0])))
    
    for pair in tqdm(zip(list_msa_seq, list_trees)):
        
        # Read msa and tree
        msa_seq, tree = pair
        one_msa_path = msas_path / f"{msa_seq}"
        one_tree_path = trees_path / f"{tree}"
        
        msa_sequences = utils.read_msa(one_msa_path, max_depth)
        tree = Phylo.read(one_tree_path, "newick")
        
        # Create their respective distances 
        distance_matrix = data_esm.create_distance_matrix(msa_sequences, tree)
        triu_indices = np.triu_indices(distance_matrix.shape[-1], k=1)
        dists = distance_matrix[triu_indices]
        
        # Save at location
        np.save(dists_path / f"{msa_seq.split('.')[0]}.npy", dists)

if __name__ == "__main__":
    
    # Parsing command line options
    parser = ArgumentParser()
    
    # Path to the esm_folder
    parser.add_argument("-ef", '--esm_folder', action="store", dest="esm_folder", default='/content/drive/MyDrive/data/Synthetic_data=50', help='Path to the ESM generated synthetic sequences', metavar='FOLDER')
    
    # Path to the distances folder in train
    parser.add_argument('-dft', '--dists_folder_train', action='store', dest='dists_folder_train', default='/content/drive/MyDrive/data/Synthetic_data=50/train/distances')
    
    # Path to the distances folder in test
    parser.add_argument('-dfts', '--dists_folder_test', action='store', dest='dists_folder_test', default='/content/drive/MyDrive/data/Synthetic_data=50/val/distances')
    
    # Get argument
    args = parser.parse_args()
    esm_folder = pathlib.Path(args.esm_folder)
    dists_folder_train = pathlib.Path(args.dists_folder_train)
    dists_folder_test = pathlib.Path(args.dists_folder_test)
    
    # Generate paths
    train_path_alignments, train_path_trees, _, test_path_alignments, test_path_trees, _ = data_esm.create_paths(esm_folder)
    
    # Create distance matrices
    create_and_store_dists_matrices(train_path_alignments, train_path_trees, dists_folder_train)
    create_and_store_dists_matrices(test_path_alignments, test_path_trees, dists_folder_test)    