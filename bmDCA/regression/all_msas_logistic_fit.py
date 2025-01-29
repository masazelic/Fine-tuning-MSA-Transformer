# Script for 
# Start with importing libraries
import os
import pathlib
import itertools
import string
from typing import List, Tuple
import warnings

import tqdm

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

from patsy import dmatrices
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import statsmodels.api as sm

import esm
import torch

from Bio import SeqIO
from Bio import Phylo

def perform_regressions_msawise(pfam_families_train, approach, normalize_dists=False):
    """ Perfroms regression on all MSA sequences. """
    df = pd.DataFrame(columns=[f"lyr{i}_hd{j}" for i in range(n_layers) for j in range(n_heads)] + ["dist"], dtype=np.float64)

    for pfam_family in pfam_families_train:
        dists = np.load(DISTS_FOLDER / f"{pfam_family}_{approach}.npy")
        attns = np.load(ATTNS_FOLDER / f"{pfam_family}_{approach}_mean_on_cols_symm.npy")

        if normalize_dists:
            dists = dists.astype((np.float64))
            dists /= np.max(dists)
            
        triu_indices = np.triu_indices(attns.shape[-1])
        attns = attns[..., triu_indices[0], triu_indices[1]]
        dists = dists[triu_indices]
        df2 = pd.DataFrame(attns.transpose(2, 0, 1).reshape(-1, n_layers * n_heads),
                           columns=[f"lyr{i}_hd{j}" for i in range(n_layers) for j in range(n_layers)])
        df2["dist"] = dists
        df = pd.concat([df, df2], ignore_index=True)

    # Carve out the training matrices from the training and testing data frame using the regression formula
    formula = "dist ~ " + " + ".join([f"lyr{i}_hd{j}" for i in range(n_layers) for j in range(n_heads)])
    y_train, X_train = dmatrices(formula, df, return_type="dataframe")

    # Fit the model
    binom_model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
    binom_model_results = binom_model.fit(maxiter=200, tol=1e-9)

    regr_results_hamming_common = {}
    for pfam_family in pfam_families:
        dists = np.load(DISTS_FOLDER / f"{pfam_family}_{approach}.npy")
        attns = np.load(ATTNS_FOLDER / f"{pfam_family}_{approach}_mean_on_cols_symm.npy")
        
        if normalize_dists:
            dists = dists.astype((np.float64))
            dists /= np.max(dists)
            
        depth = len(dists)

        triu_indices = np.triu_indices(depth)
        attns = attns[..., triu_indices[0], triu_indices[1]]
        dists = dists[triu_indices]
        attns = attns.transpose(2, 0, 1).reshape(-1, n_layers * n_heads)

        df = pd.DataFrame(attns,
                          columns=[f"lyr{i}_hd{j}" for i in range(n_layers) for j in range(n_heads)])
        df["dist"] = dists
        _, X = dmatrices(formula, df, return_type="dataframe")

        y_pred = binom_model_results.predict(X).to_numpy()

        regr_results_hamming_common[pfam_family] = {
            "bias": binom_model_results.params[0],
            "coeffs": binom_model_results.params.to_numpy()[-n_layers * n_heads:].reshape(n_layers, n_heads),
            "y": dists,
            "y_pred": y_pred,
            "depth": depth,
        }

    return regr_results_hamming_common

def create_dist_comparison_mat(y, y_pred, n_rows):
    assert len(y) == len(y_pred)

    comparison_mat = np.zeros((n_rows, n_rows), dtype=np.float32)
    ct = 0
    for i in range(n_rows):
        for j in range(i, n_rows):
            # Order is important as we want the diagonal to be a prediction
            comparison_mat[i, j] = y[ct]
            comparison_mat[j, i] = y_pred[ct]
            ct += 1
    assert ct == len(y)

    return comparison_mat

if __name__ == "__main__":
    
    # Plotting settings
    SMALL_SIZE = 50
    MEDIUM_SIZE = 60
    BIGGER_SIZE = 70
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["times"],
        "font.size": MEDIUM_SIZE,
        "axes.titlesize": BIGGER_SIZE,
        "axes.labelsize": BIGGER_SIZE,
        "figure.titlesize": BIGGER_SIZE,
        "xtick.labelsize": MEDIUM_SIZE,
        "ytick.labelsize": MEDIUM_SIZE,
        "legend.fontsize": MEDIUM_SIZE
    })
    
    # Setting random seed for fixed performance
    SEED = 42
    rng = np.random.default_rng(SEED)
    
    DISTS_FOLDER = pathlib.Path("./distance_matrix")
    ATTNS_FOLDER = pathlib.Path("./col_attentions_subtree")
    HAMM_FOLDER = pathlib.Path("./hamming_random")
    
    print(DISTS_FOLDER)
    print(ATTNS_FOLDER)

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
    approach = 'subtree'
    MAX_DEPTH = 600

    n_layers = n_heads = 12
    
    pfam_families_train = pfam_families[:12]
    regr_results_hamming_common = perform_regressions_msawise(pfam_families_train, approach=approach, normalize_dists=True)
    
    cmap = cm.bwr
    vpad = 30
    x_vals_coeffs = np.arange(0, n_heads, 2)
    y_vals_coeffs = np.arange(0, n_layers, 2)
    fig, axs = plt.subplots(figsize=(43, 10),
                            nrows=1,
                            ncols=5,
                            gridspec_kw={"width_ratios": [10, 3, 10, 10, 10]},
                            constrained_layout=True)

    coeffs = regr_results_hamming_common[pfam_families[0]]["coeffs"]
    im = axs[0].imshow(coeffs, norm=colors.CenteredNorm(), cmap=cmap)
    cbar = fig.colorbar(im, ax=axs[0], fraction=0.05, pad=0.03)
    axs[0].set_xticks(x_vals_coeffs)
    axs[0].set_yticks(y_vals_coeffs)
    axs[0].set_xticklabels(list(map(str, x_vals_coeffs + 1)))
    axs[0].set_yticklabels(list(map(str, y_vals_coeffs + 1)))

    axs[1].plot(np.mean(np.abs(coeffs), axis=1),
                np.arange(n_layers),
                "-o",
                markersize=12,
                lw=5)
    axs[1].invert_yaxis()
    axs[1].set_yticks(y_vals_coeffs)
    axs[1].set_yticklabels(list(map(str, y_vals_coeffs + 1)))
    axs[1].set_xticks([0, 10, 20])

    axs[0].set_title("Regression coefficients", pad=vpad)
    axs[1].set_title("Avg.\ abs.\ coeff.", pad=vpad)

    for i, pfam_family in enumerate(pfam_families[-3:]):
        y = regr_results_hamming_common[pfam_family]["y"]
        y_pred = regr_results_hamming_common[pfam_family]["y_pred"]
        n_rows = regr_results_hamming_common[pfam_family]["depth"]

        hamming_comparison = create_dist_comparison_mat(y, y_pred, n_rows)
        axs[2 + i].imshow(np.triu(hamming_comparison, k=1) + np.tril(np.full_like(hamming_comparison, fill_value=np.nan)),
                        cmap="Blues",
                        vmin=0,
                        vmax=1)
        pos = axs[2 + i].imshow(np.tril(hamming_comparison) + np.triu(np.full_like(hamming_comparison, fill_value=np.nan), k=1),
                                cmap="Greens",
                                vmin=0,
                                vmax=1)

        axs[2 + i].set_xlabel("Sequence")

        axs[2 + i].set_title(pfam_family, pad=vpad)

    axs[0].set_ylabel("Layer")
    axs[0].set_xlabel("Head")
    axs[1].set_ylabel("Layer")
    axs[2].set_ylabel("Sequence")

    plt.savefig('all_msas_random.png')

    df_regr_results_hamming_common = pd.DataFrame()

    fig, axs = plt.subplots(figsize=(25, 15),
                            nrows=3,
                            ncols=5,
                            sharex=True,
                            sharey=True,
                            constrained_layout=True)

    for i, pfam_family in enumerate(pfam_families):
        df_regr_results_hamming_common.loc[pfam_family, "Depth"] = regr_results_hamming_common[pfam_family]["depth"]
        y = regr_results_hamming_common[pfam_family]["y"]
        y_pred = regr_results_hamming_common[pfam_family]["y_pred"]
        n_samples = len(y)
        df_regr_results_hamming_common.loc[pfam_family, "RMSE"] = np.linalg.norm(y - y_pred) / np.sqrt(n_samples)
        y_std = np.std(y)
        df_regr_results_hamming_common.loc[pfam_family, "Std"] = y_std
        pearson = pearsonr(y, y_pred)[0]
        df_regr_results_hamming_common.loc[pfam_family, "Pearson"] = pearson
        slope = pearson * y_std / np.std(y_pred)
        df_regr_results_hamming_common.loc[pfam_family, "Slope"] = slope
        df_regr_results_hamming_common.loc[pfam_family, "R^2"] = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        intercept = np.mean(y) - slope * np.mean(y_pred)
        
    print(df_regr_results_hamming_common[12:])