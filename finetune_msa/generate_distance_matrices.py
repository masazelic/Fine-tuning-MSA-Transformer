import os
import sys

import pathlib
import numpy as np
import string
import itertools
from itertools import combinations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Phylo

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

def collect_sequences(msa_family, subtree, MAX_DEPTH):
    """ Go over the subtree and collect all the sequences in a list. """
    
    sequences = []
    for leaf in subtree.get_terminals():
        name = 'seq' + str(leaf.name)
        
        # I want to save both sequence name and sequence and save it in .fasta to make code easier to reproduce
        for seq in msa_family:
            if name == seq[0]:
                sequences.append(seq)

    return sequences[:MAX_DEPTH]

def parse_sample_name(sequence_name):
    """ Function to parse synthetic MSA sequence name since it is in the format sampleNumber. """
    return sequence_name.split('seq')[1]

def create_distance_matrix(msa_sequences, tree):
    """ Provided MSA sequences (their names) and tree create distance matrix. """
    num_sequences = len(msa_sequences)

    # Create distance matrix that will store these 
    distance_matrix = np.zeros((num_sequences, num_sequences))

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            # Calculate distance between sequence pairs using tree
            distance = tree.distance(parse_sample_name(msa_sequences[i][0]), parse_sample_name(msa_sequences[j][0]))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix
    
    return distance_matrix

def save_to_fasta(sequence_list, output_path):
    """ Save a list of ('sequence_name', 'sequence') tuples to fasta. """
    records = []
    for seq_name, sequence in sequence_list:
        # Create a SeqRecord of each tuple
        record = SeqRecord(Seq(sequence), id=seq_name, description="")
        records.append(record)

    # Write the list of SeqRecords to a fasta file
    SeqIO.write(records, output_path, "fasta")

def find_subtree(tree_path):
    """ Function for finding the first subtree with fewer than 500 terminal nodes (leaves). """
    
    # Load the phylogenetic tree
    tree = Phylo.read(tree_path, "newick")

    # Start the recursive search from the root of the tree
    subtree, num_leaves = find_subtree_recur(tree.root)

    if subtree:
        print(f"The first subtree with fewer than 500 terminal nodes has {num_leaves} leaves.")

        # We need to return that subtree so that we can obtain the distance matrix
        return subtree

    else:
        print("No subtree found with fewer than 500 terminal nodes.")

def find_subtree_recur(clade):
    """ Function that recursively goes over subtrees in search for the first one with less than 600 leaves. """
    
    # Get the terminal nodes (leaves) of the current clade
    terminals = clade.get_terminals()

    # Check if the current clade has fewer than 500 leaves
    if ((len(terminals) > 400) and (len(terminals) < 500)):
        return clade, len(terminals)

    # If not, recursively search the child clades
    for child in clade.clades:
        result, num_leaves = find_subtree_recur(child)
        if result: # If a valid subtree is found in the child return it
            return result, num_leaves

    # If no subtree with fewer than 500 leaves is found, return None
    return None, 0

def random_subsampling(msa, MAX_DEPTH):
    """ Randomly taking MAX_DEPTH of synthetic MSA sequences. """
    # Randomly generate the indexes of the sequences that will be sampled
    np.random.seed(42) # Fix seed for the reproducibility
    indexes = np.random.randint(low=0, high=len(msa), size=MAX_DEPTH)

    # Take the sequences corresponding to those indexes
    return [msa[i] for i in indexes]

def subsampling(MSA_FOLDER, TREE_FOLDER, MAX_DEPTH, OUTPUT_MSA_FOLDER, DM_FOLDER, pfam_family, approach):
    """ Subsampling synthetic MSAs with different approaches for a signle family. """

    # Define all the paths
    tree_path = TREE_FOLDER / f"{pfam_family}_modified.newick"
    backup_tree_path = TREE_FOLDER / f"{pfam_family}_full_no_gapped_modified.newick"
    msa_path = MSA_FOLDER / f"{pfam_family}.fasta"

    # Read MSA
    msa = read_msa(msa_path, 100000)

    # If we are generating from subtree
    if approach == "subtree":

        # Tree files have 2 different names - check the one or the other
        if os.path.exists(tree_path):
            tree = find_subtree(tree_path)

        elif os.path.exists(backup_tree_path):
            tree = find_subtree(backup_tree_path)

        # If there is more than 500 sequences in subtree, subsample
        max_depth_seq = collect_sequences(msa, tree, MAX_DEPTH)

        # Create distance matrix
        distance_matrix = create_distance_matrix(max_depth_seq, tree)
    
        # Write MSAs to fasta and distance matrix to npy
        print(f"The random MSA for family {pfam_family} has been obtained.")
        save_to_fasta(max_depth_seq, OUTPUT_MSA_FOLDER / f"{pfam_family}_{approach}.fasta")
        np.save(DM_FOLDER / f"{pfam_family}_{approach}.npy", distance_matrix)

    # If we are generating by randomly sampling 
    elif approach == "random":

        # Random subsampling
        max_depth_seq = random_subsampling(msa, MAX_DEPTH)
        print(len(max_depth_seq))

        # Load the tree
        if os.path.exists(tree_path):
            tree = Phylo.read(tree_path, "newick")
        
        elif os.path.exists(backup_tree_path):
            tree = Phylo.read(backup_tree_path, "newick")

        # Create distance matrix
        distance_matrix = create_distance_matrix(max_depth_seq, tree)

        # Write MSAs to fasta and save distance matrix
        save_to_fasta(max_depth_seq, OUTPUT_MSA_FOLDER / f"{pfam_family}_{approach}.fasta")
        np.save(DM_FOLDER / f"{pfam_family}_{approach}.npy", distance_matrix)

        # Write comment so that we can track the process 
        print(f"MSA for {pfam_family} generated.")


if __name__ == "__main__":

    # Parsing command-line options
    parser = ArgumentParser()

    # Path to the tree folder
    parser.add_argument("-ft", "--tree_folder", action="store", dest="tree_folder", default="/content/drive/MyDrive/data/trees", help="Write path to tree folder.", metavar="FOLDER")

    # Path to the MSA folder
    parser.add_argument("-mf", "--msa_folder", action="store", dest="msa_folder", default="/content/drive/MyDrive/data/msa_synthetic", help="Write path to MSA folder.", metavar="FOLDER")

    # Set MAX_DEPTH
    parser.add_argument("-md", "--max_depth", action="store", dest="max_depth", default='100', help="Write MAX_DEPTH of MSA that will be analyzed for distance.", type=int)

    # Set subsampling approach
    parser.add_argument("-a", "--subsampling_approach", action="store", dest="subsampling_approach", default="subtree", help="Write subsampling approach. Possible 'subtree' or 'random'.")

    # Path to the output MSA Folder
    parser.add_argument("-om", "--output_msa_folder", action="store", dest="output_msa", default="/content/drive/MyDrive/data/subsampled_msa", help="Write path to output MSA folder.")

    # Path to the output Distance Matrix Folder
    parser.add_argument("-dm", "--distance_matrix_folder", action="store", default="/content/drive/MyDrive/data/distance_matrix", dest="distance_matrix_folder", help="Write path to Distance Matrix folder.")

    # Get arguments 
    args = parser.parse_args()
    TREE_FOLDER = pathlib.Path(args.tree_folder)
    MSA_FOLDER = pathlib.Path(args.msa_folder)
    MAX_DEPTH = args.max_depth
    approach = args.subsampling_approach
    OUTPUT_MSA_FOLDER = pathlib.Path(args.output_msa)
    DM_FOLDER = pathlib.Path(args.distance_matrix_folder)

    pfam_families = ["PF00004", "PF00005", "PF00041", "PF00072", "PF00076", "PF00096", "PF00153", "PF00271", "PF00397", "PF00512", "PF00595", "PF01535", "PF02518","PF07679", "PF13354"]

    # If folders for storing MSAs and distance matrices don't exist create them 
    for folder in [OUTPUT_MSA_FOLDER, DM_FOLDER]:
        if not folder.exists():
            os.mkdir(folder)

    # Iterate over every family to create its sumbsampled MSA and distance matrix
    for family in pfam_families:
        subsampling(MSA_FOLDER, TREE_FOLDER, MAX_DEPTH, OUTPUT_MSA_FOLDER, DM_FOLDER, family, approach)

    