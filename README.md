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