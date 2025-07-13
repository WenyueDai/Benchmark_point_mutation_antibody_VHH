# TEMPRO
Nanobody Melting Temperature Prediction using Protein Embeddings

We present a method for estimating a nanobody’s melting temperature with their protein structure embeddings as predictive features of thermostability using an input fasta file.


**Files:**

1) embedding_generator.zip - contains the code for generating embeddings using ESM and fasta file of all 567 nanobodies used in the study.
   
2) paper_results.zip - contains the codes and necessary datasets to reproduce our results from the paper.
   
3) user.zip - contains our pre-trained models for predicting your own nanobody thermostability.

4) tm_predictors.rar - contains the supplemental files for the TEMPRO.ipynb notebook if you want to run the last cell/s. You can also generate your own embeddings using the embedding_generator.zip folder. (Note: I could not upload the 15B embeddings due to the file size. I was able to upload 650M and 3B only.)



**Instructions:**

To predict your own nanobody melting temperature using its protein sequence:

1) Please install the required packages using the requirements.txt file such as:
   
   'conda install -r requirements.txt'
   
   or
   
   'pip install -r requirements.txt.'
   
3) Navigate to the user folder, open "TEMPRO.ipynb" Jupyter Notebook, and run the codes.


Upon use, please cite our paper:
Alvarez, J.A.E., Dean, S.N. TEMPRO: nanobody melting temperature estimation model using protein embeddings. Sci Rep 14, 19074 (2024). https://doi.org/10.1038/s41598-024-70101-6

Note for Keras models: it is important to install the correct Python version (in the requirements.txt file; version 3.9) to properly load the Keras models for predicting nanobody thermostability.

Notes for reproducing paper results (inside paper_results.zip file):
1) "sdab_data.xlsx" is the processed dataset containing all 567 single-domain antibodies used for the analyses.
2) "raw_nbthermo_json_data.csv" contains the raw sdab data from NbThermo database.
3) "raw_nrl_data.xlsx" contains the raw sdab data manually curated by the authors.

