import numpy as np
import pathlib
import torch

def create_train_test_sets_per_family(attns, dists, train_idx, normalize_dists=False, ensure_same_size=False, zero_attention_diagonal=False):
    """ 
    Attentions assumed averaged across column dimensions, i.e. 4D tensors. Creates train-test split for specific family.
    
    Args:
        attns (np.array): Column attentions for the specific family.
        dists (np.array): Distance matrix.
        train_idx (np.array): Indexes of sequences that go in train pool.
        normalize_dists (bool): Whether to normalize distances with maximum distance.
    
    Returns:
        attns_train (np.array): Embeddings of size 144 x number of pairs of train sequences.
        dists_train (np.array): Distances between each pair of train sequences. 
        attns_test (np.array): Embeddings of size 144 x number of pairs of test sequences. 
        dists_test (np.array): Distances between each pair o test sequences. 
        n_rows_train (int): Number of train sequences.
        n_rows_test (int): Number of test sequences.
    
    """
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

def merge_all_families(pfam_families, dists_folder, attns_folder, train_indexes, approach, normalize_dists=True, ensure_same_size=False, zero_attention_diagonal=False):
    """
    For loading all the families. 
    
    Args:
        pfam_family (list): List containing all the family names.
        dists_folder (path.Pathlib): Path to the folder where distances are stored.
        attns_folder (path.Pathlib): Path to the folder where attentions are stored.
        train_indexes (np.array): Indexes that will be part of train set from all families. 
        approach (str): Either "random" or "subtree".
        normalize_dists (bool): Whether to normalize distance.
        
    Returns:
        attns_train (torch.Tensor): Train embeddings for all families.
        dists_train (torch.Tensor): Train distances for all families.
        attns_test (torch.Tensor): Test embeddings for all families. 
        dists_test (torch.Tensor): Test distances for all families. 
    """

    attns_train_random = []
    attns_test_random = []
    dists_train_random = []
    dists_test_random = []

    for i, pfam_family in enumerate(pfam_families):
        
        # Load path to the specific column attentions and distances
        dists_random = np.load(dists_folder / f"{pfam_family}_{approach}.npy")
        attns_random = np.load(attns_folder / f"{pfam_family}_{approach}_mean_on_cols_symm.npy")

        # Load data
        ((attns_train_r, dists_train_r), (attns_test_r, dists_test_r), (n_rows_train_r, n_rows_test_r)) = create_train_test_sets_per_family(attns_random, dists_random, train_indexes[i], normalize_dists=normalize_dists, ensure_same_size=ensure_same_size, zero_attention_diagonal=zero_attention_diagonal)
        
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

