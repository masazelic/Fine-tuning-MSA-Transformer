# Inferring phylogenetic relationships using MSA Transformer derived embeddings

**Lab Immersion at EPFL**  
**Lab:** Bitbol Lab - Laboratory of Computational Biology and Theoretical Biophysics  
**Professor:** Anne-Florence Bitbol  
**Supervisor:** Damiano Sgarbossa  

## Description

The project is extension of paper [Protein language models trained on multiple sequence alignments learn phylogenetic relationships](https://doi.org/10.1038/s41467-022-34032-y) which shows that Regression on predictors derived from MSA Transformer column attention's are able to capture Hamming distance, which is simple proxy of phylogenetic relationship. However, without the **true** phylogenetic tree, we lack an 'accurate' ground truth values for distances.  

Therefore, we go beyond this by *generating synthetic sequences* along the existing or known tree and using Patristic distance derived from the tree to analyze if MSA Transformer based embeddings can capture them. We do this by repeating the Regression analysis, but also by fine-tuning MSA Transformer. 

## Getting Started

### Installation

1. Clone the repository. 
```
git clone https://github.com/masazelic/Fine-tuning-MSA-Transformer.git
```

2. Install the requirements. 
```
pip install -r requirements.txt
```

### bmDCA Approach

For running following section of code, go into bmDCA folder.  

1. Subsampling Data

Synthethic sequences for 15 protein families (PF00004, PF00005, PF00041, PF00072, PF00076, PF00096, PF00153, PF00271,PF00397, PF00512, PF00595, PF01535, PF02518, PF07679, PF13354) were generated using bmDCA approach along the trees corresponding to the natural sequences. Trees can be found at [link](https://drive.google.com/drive/folders/1zO5LwJENLHyX10qNC-xCRmsmbAWE8bLL?usp=drive_link), while sequences can be found at [link](https://drive.google.com/drive/folders/1BELhdgIYErX-Gfkr0gBtNpj1f4JqcQ7B?usp=drive_link).  

If you want to subsample 100 sequences per family through *random* subsampling approach, run:

```
python generate_distance_matrices.py -ft <your_tree_folder> -mf <your_msa_folder> -md 100 -a random -om <your_msa_output_folder> -dm <your_distance_matrix_folder>
```

NOTE: For subsampling 100 random sequences per family you will need around 47 minutes for all families. If you increase it to 500 it will take over 2 hours for one. 

For more details on available command line arguments, run:

```
python generate_distance_matrices.py -h
```