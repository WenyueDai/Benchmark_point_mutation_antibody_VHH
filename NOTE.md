(Keep updating 20250706)

Ablang2
conda activate efficient-evolution

Antifold
1. Antifold env is installed based on github, and also pip install gemmi to convert pdb to cif
2. When running antifold, the subprocess is called to run antifold_worker.py in antifold environment

ESM2
3. esm environment is created (for only esm2 and esm-1f): 
conda env create -f esm_env.yml & conda activate esm

Antiberty
4. conda create -n antiberty python=3.9
conda activate antiberty
conda install pandas
pip install antiberty
5. Go to /home/eva/miniconda3/envs/antiberty/lib/python3.9/site-packages/antiberty/AntiBERTyRunner.py 
Change only this line (at the very end of pseudo_log_likelihood) from:

python
Copy
Edit
labels = self.tokenizer.encode(
    " ".join(list(s)),
    return_tensors="pt",
)[:, 1:-1]
to

python
Copy
Edit
labels = self.tokenizer.encode(
    " ".join(list(s)),
    return_tensors="pt",
).to(self.device)[:, 1:-1]

6. test with from antiberty import AntiBERTyRunner
antiberty = AntiBERTyRunner()
sequences = ["EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS"]
pll = antiberty.pseudo_log_likelihood(sequences, batch_size=1)
print(pll)

7. Pseudo log-likelihood
To use AntiBERTy to calculate the pseudo log-likelihood of a sequence, use the pseudo_log_likelihood function. The pseudo log-likelihood of a sequence is calculated as the average of per-residue masked log-likelihoods. The output is a list of pseudo log-likelihoods, corresponding to the input sequences.
Caplacizumab  -> -1.02
This means “on average, when masking each position one at a time, the model could reconstruct the original with PLL -1.02.”