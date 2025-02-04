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

**NOTE**: For subsampling 100 random sequences per family you will need around 47 minutes for all families. If you increase it to 500 it will take over 2 hours for one. 

For more details on available command line arguments, run:

```
python generate_distance_matrices.py -h
```

2. Regression 

For performing regression analysis, go over `Regression.ipynb` in `regression` folder. Cell that requires user input of specific paths/parameters is clearly noted.  

3. Training Fully-connected network with MSA Transformer's column attentions  

Code supporting this approach can be found in `fc_network` folder. 

For tuning hyperparameters such as network architecture and learning rate for *subtree* approach use the follwing command (you need to enter the folder before doing so):

```
python hyperparameter_tuning.py -af <your_attentions_folder> -df <your_distances_folder> -a <subtree/random>
```

**NOTE 1:** The code assumes that attention matrices and distances are pre-computed. For the specific families noted above, you can download patristic distances from [link](https://drive.google.com/drive/u/0/folders/11wxuhhqMeEoEmp_EJYjiIYCOFJ-vZqyY) and attention matrices from [link](https://drive.google.com/drive/u/0/folders/19jG7KDS7E8LqrDNSzA6pAEJAzDVXWfLF) (subtree approach) and [link](https://drive.google.com/drive/u/0/folders/1EYuajAN9sAv6sGH8N77wHv3OlcbhtpJs).  

**NOTE 2:** Notation for both distances and attention matrices is in form *familiyName_approach_var.npy*.  

Running this file produces `.pkl` file containing best selected hyperparameters based on the criterion. You can find best hyperparameters for both approaches in `parameters` folder.  

For **training** the model with the optimal parameters, you can run following command:

```
python run.py -af <your_attentions_folder> -df <your_distances_folder> -a <subtree/random> -r train -bp <your_best_params_file>
```  

Running this command results in model named as `trained_approach.pth` being saved in `models` folder. Trained models for both approaches can already be found there.  

For **evaluating** the trained model on test dataset, you can run following command:

```
python run.py -af <your_attentions_folder> -df <your_distances_folder> -a <subtree/random> -r test -bp <your_best_params_file>
```  





