(Keep updating 20250706)

point mutation
conda create -n point_mutation python=3.9 -y
conda activate point_mutation
conda install -c conda-forge pandas pytorch -y
pip install abnumber antiberty ablang2 ImmuneBuilder
conda install seaborn matplotlib -y

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

Nanobert
conda activate antiberty
pip install protobuf
conda install -c conda-forge protobuf

from huggingface_hub import snapshot_download
snapshot_download("NaturalAntibody/nanoBERT")
ls ~/.cache/huggingface/hub/


pyrosetta
pyrosetta env: https://github.com/WenyueDai/Solublize_Transmembrane_Protein


LM-design
# clone project
git clone https://url/to/this/repo/ByProt.git
cd ByProt

# create conda virtual environment
env_name=ByProt

conda create -n lm_design python=3.7 pip
conda activate lm_design

# automatically install everything else
bash install.sh

Download checkpoint file from lm_design_esm2_650m and put it into correct path

pip install fair-esm
pip install lmdb
pip install tmtools

go to src/byprot/models/fixedbb/lm_design/esm2_adapter.py
change tied_pos_list = prev_decoder_out["tied_pos_list"] to tied_pos_list = prev_decoder_out.get("tied_pos_list", None)
(because we are not constraint any)

Need to modify designer.py (src/byprot/tasks/fixedbb/designer.py)
replace calculate_metrics() with this:
def calculate_metrics(self):
    native_seq = self._structure['seq']
    output_tokens = self._predictions.output_tokens
    output_scores = self._predictions.output_scores  # shape: (B, L)

    results = []

    for i, prediction in enumerate(output_tokens):
        rec = np.mean([(a == b) for a, b in zip(native_seq, prediction)])
        print(f"prediction: {prediction}")
        print(f"recovery: {rec}")

        scores = output_scores[i].detach().cpu().numpy() # per-residue log-probabilities (usually in log space)
        avg_nll = -np.mean(scores)
        perplexity = np.exp(avg_nll)

        print(f"perplexity: {perplexity}")
        print()

        results.append({
            "predicted_seq": prediction,
            "recovery": rec,
            "perplexity": perplexity,
            "avg_nll": avg_nll,
            "length": len(scores),
        })

    return results[0]  # return the first one if batch size is 1

change original set_structure
def set_structure(
            self, 
            pdb_path, 
            chain_list=[], 
            masked_chain_list=None, 
            verbose=False
        ):
        from pathlib import Path
        pdb_id = Path(pdb_path).stem

        print(f'loading backbone structure from {pdb_path}.')
        
        parsed = self.data_processor.parse_PDB(
            pdb_path, 
            input_chain_list=chain_list, 
            masked_chain_list=masked_chain_list
        )

        # Convert tuple to dict format if needed
        if isinstance(parsed, tuple):
            coords, native_seq = parsed
            self._structure = {
                "coords": coords,
                "seq": native_seq
            }
        elif isinstance(parsed, dict):
            self._structure = parsed
        else:
            raise TypeError(f"Unexpected return type from parse_PDB: {type(parsed)}")

        if verbose:
            print("DEBUG: Structure keys:", self._structure.keys())
            return self._structure