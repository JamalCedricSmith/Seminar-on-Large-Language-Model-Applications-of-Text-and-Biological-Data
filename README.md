# Reproducing "Biological Structure and Function Emerge from Scaling Unsupervised Learning"

This repository contains the code and experimental results for a reproduction of the paper
**"Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences"**
by Rives et al. (2021).

We implemented the algorithms and linear probing techniques using the fair-esm package to replicate the
experimental setup and validate the reported results. We adopted an **Inference-Only Strategy**, utilizing
official pre-trained weights without further unsupervised pretraining.

## Setup & Methodology

### Environment
Experiments were conducted using a Google Colab environment with GPU acceleration (Tesla T4 / P100).

### Package
fair-esm (Version 2.0.0)

### Reproducibility
A fixed random seed of `42` was applied to all operations.

### Model Substitution
For qualitative analyses requiring the unpublished 36-layer UniParc development model, we substituted it
with `esm1-t34-670M-UR100`. While this model uses a different masking objective, it remains the closest
open-source approximation to the original scale.


## Reproduction Results

### Experiment 1: Biochemical Properties (t-SNE)

**Objective**  
Investigate whether the model explicitly organizes biochemical properties without supervision by
projecting final embedding weights into 2D space.

**Original Claim**  
The embedding space reflects biochemical interchangeability with distinct clustering of hydrophobic,
polar, aromatic, and charged residues.

**Our Reproduction**  
Consistent with the original study. Distinct clusters for hydrophobic, aromatic, and small amino acids
are observed. Organization by molecular weight and charge is preserved. Histidine (H) aligns between
Arginine (R) and Asparagine (N), reflecting shared biochemical properties.


### Experiment 2: Biological Variations (PCA & t-SNE)

**Objective**  
Analyze whether the representation space captures evolutionary structure using orthologous genes across
diverse species.

**t-SNE Results**  
Successfully reproduced. Training induces significant structural organization. Unigram baselines show
diffuse clouds, while the trained model separates orthologous groups into dense clusters.

**PCA Results**  
Failed to reproduce. The original paper reported linear disentanglement where species and orthology
appeared as orthogonal axes. Despite testing multiple COGs and species combinations, this separation
could not be reproduced on the first two PCA axes.


### Experiment 3: Remote Homology Detection

**Objective**  
Evaluate whether structural similarity is encoded in the embedding space using the SCOPe dataset,
benchmarked against HHblits.

**Methodology**  
A rigorous SCOPe split was used to test generalization across **Family**, **Superfamily**, and **Fold**
levels.

**Our Reproduction**  
Using ESM-1b, higher Hit-10 scores at the Superfamily level were obtained compared to those reported in
the original paper.

Model           Dataset   Hit-10(SF)  Hit-10(Fold)  AUC(SF)  AUC(Fold)
Transformer-6   UR50/S    0.7442      0.5379        0.8468   0.6751
ESM-1b          UR50/S    0.8790      0.4719        0.8621   0.6361


### Experiment 4: Alignment within Protein Families

**Objective**  
Test whether the model implicitly learns evolutionary structural information typically captured by MSAs
by comparing cosine similarity distributions of aligned versus unaligned residue pairs from the Pfam
database.

**Our Reproduction**  
A clear shift between cosine similarity distributions of aligned and unaligned residue pairs is observed.
This confirms that the model captures evolutionary correspondence even when amino acid identities differ,
isolating structural context from lexical similarity.


### Experiment 5: Secondary & Tertiary Structure Prediction

**Objective**  
Train linear probes on frozen embeddings to predict structural features.


#### Secondary Structure Prediction (8-class)

Multi-class logistic regression models were trained on final hidden representations.

**Findings**  
ESM-1b consistently achieves the highest performance. Accuracy remains stable across Family,
Superfamily, and Fold splits, indicating strong generalization. Models trained on diverse datasets
(UR50/S, UR50/D) outperform those trained on high-density, low-diversity data (UR100).

Model           Dataset   Family        Superfamily    Fold
Transformer-6   UR50/S    62.6 ± 0.01   62.0 ± 0.01    62.1 ± 0.01
Transformer-34  UR100     65.0 ± 0.01   64.5 ± 0.01    64.4 ± 0.01*
ESM-1b          UR50/S    70.72 ± 0.00  70.2 ± 0.01    70.53 ± 0.01*

*Restricted cross-validation due to computational constraints.*


#### Tertiary Contact Prediction (Top-L Precision)

Linear projections `P` and `Q` were fit on pairwise representations to predict binary contacts
(< 8 Å).

**Findings**  
ESM-1b embeddings achieve the best overall performance. Most models underperform relative to the
original paper, with ESM-1b being the sole model that outperforms the reported precision values.

Model           Dataset   Family        Superfamily    Fold
Transformer-6   UR50/S    18.9 ± 0.01   16.6 ± 0.03    15.9 ± 0.01
Transformer-34  UR50/S    30.6 ± 0.02   24.3 ± 0.04*   25.48 ± 0.05*
ESM-1b          UR50/S    50.14 ± 0.03  45.6 ± 0.04    44.48 ± 0.02*


## References

Rives et al., "Biological structure and function emerge from scaling unsupervised learning to
250 million protein sequences", PNAS, 2021.

Meta Research (FAIR), "ESM: Evolutionary Scale Modeling", GitHub Repository.
