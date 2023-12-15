# CellGiQ

## Overview

CellGiQ, a novel framework for deciphering ligand-receptor-mediated cell-cell communication by incorporating machine learning and a quartile scoring strategy from single-cell RNA sequencing data. CellGiQ accurately inferred intercellular communication within human HNSCC tissues. CellGiQ is anticipated to dissect cellular crosstalk and signal pathways at
single cell resolution.

![Overview](Overview.png)

## Environment

- python == 3.8.13

### packages:

- tensorflow == 2.10.0
- keras == 2.10.0
- GBNN == 0.0.2

- interpret == 0.2.7

- scikit-learn == 0.24.0

- lightgbm == 3.3.5
- wheel == 0.37.1
- pands == 1.5.0
- numpy == 1.24.2

## Data

1.Data is available at [uniprot](https://www.uniprot.org/), [GEO](https://www.ncbi.nlm.nih.gov/geo/).

2.Feature extraction website at  [BioTriangle](http://biotriangle.scbdd.com/)

## Usage

1. We obtain ligand and receptor feature at  [BioTriangle](http://biotriangle.scbdd.com/)

2. Run the model to obtain the LRI, or the user-specified LRI database		

   ```
   python code/CellGiQ.py
   ```

3. Using quartile method (including Expression thresholding, Expression product and Specific expression), the cell-cell communication matrix was finally obtained.		

     ```
     python code/case study
     ```
## Change database

  If you want to test other tumors, Just replace the code in `case study.py`, `GSE103322.csv` (Note: use the specified database to replace the dataset`LRI_dataset.csv`)

## Cell-cell communication tools for comparative analysis



[CellChat](https://github.com/sqjin/CellChat)   [iTALK](https://github.com/Coolgenome/iTALK)   [LIANA](github.com/saezlab/liana)   [CellPhoneDB](https://github.com/Teichlab/cellphonedb)   [NATMI](https://github.com/asrhou/NATMI)



