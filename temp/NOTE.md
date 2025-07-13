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



ESM1F
conda create -n esm1f python=3.9 -y
conda activate esm1f
# Base scientific stack
conda install numpy=1.23 pandas=1.5 -c conda-forge -y

# Biotite from conda-forge for compatibility
conda install biotite -c conda-forge -y

# numexpr explicitly for pandas compatibility
conda install numexpr=2.8 -c conda-forge -y

# PyTorch (adjust for your GPU setup if needed)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y  # Or cuda-compatible version

# Install ESM from pip (latest)
 pip install git+https://github.com/facebookresearch/esm.git@main

python -c "import numpy; import pandas; import biotite.structure; import numexpr; print('All OK')"     
conda install scipy -y

(esm1f) eva@LAPTOP-S2RAAEJ3:~/0_point_mutation$ conda activate esm1f
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
2.5.1
None

For CPU only
pip install torch-scatter torch-sparse torch-geometric \
  -f https://data.pyg.org/whl/torch-2.5.1+cpu.html


Observation:
Left: Unrelaxed structure
Right: PyRosetta-relaxed structure
The key difference is how PyRosetta's relaxed structure affects agreement between PyRosetta and other models:

Model Pair	Correlation (Unrelaxed)	Correlation (Relaxed)	Change
PyRosetta vs Ablang	0.097	0.11	⬆ slight increase
PyRosetta vs Antifold	0.20	0.28	⬆ improved
PyRosetta vs ESM1f	0.33	0.37	⬆ improved
PyRosetta vs ESM1v	0.12	0.15	⬆ slight
PyRosetta vs Nanobert	0.081	0.099	⬆ slight

For other models (not involving PyRosetta), correlations remain similar.

Conclusions:
PyRosetta's agreement with other models improves slightly after relaxation.

This is most notable with Antifold (0.20 → 0.28) and ESM1f (0.33 → 0.37).

Suggests that relaxing the structure may reduce noise in PyRosetta’s ΔΔG calculations, making its energetic predictions more consistent with ML-based models trained on fixed backbones.

Correlations are still relatively low between PyRosetta and other models.

Even after relaxation, PyRosetta’s highest correlation is ~0.37 (with ESM1f).

This reflects the fundamental difference in modeling philosophy:

ML models predict mutational effects from sequence or minimal structure.

PyRosetta relies on energy minimization and atomistic modeling, which can diverge from statistical learning.

Inter-ML model correlations are high and unaffected by relaxation.

Ablang/ESM1v and Antifold/ESM1f correlations remain strong (>0.6–0.8), indicating robust agreement across different sequence-based or structure-informed models.

Final Thought:
Using relaxed structures marginally improves compatibility between PyRosetta and ML-based models, but they remain fundamentally distinct tools. PyRosetta may capture structural physics not easily inferred from ML embeddings, and thus can provide complementary insights, especially when physical stability is key.

What the Correlation Results Say:
1. ML models (Ablang, ESM1v, Antifold, etc.) correlate well with each other
Correlations between ML models (especially ESM1v–Ablang, Antifold–ESM1f) remain high across both unrelaxed and relaxed structures (e.g., 0.8+).

This shows internal consistency among models trained on sequence or learned embeddings.

2. PyRosetta shows low correlation with ML models
Even after structure relaxation, correlation with others stays below 0.4.

PyRosetta vs Ablang: ~0.097 → 0.11

PyRosetta vs Antifold: ~0.20 → 0.28

PyRosetta vs ESM1f: ~0.33 → 0.37

Slight improvement after relaxation, but overall still weak agreement.

3. Relaxation improves correlation marginally
You gain small but consistent increases in correlation with ML models after relaxing the structure.

This implies PyRosetta’s ΔΔG predictions are sensitive to atomic detail and clash resolution, while ML models are robust to structure noise.

Interpretation in the Context of VHH Design:
Factor	Insight
ML model agreement	Strong correlation suggests you can trust ML models (e.g. ESM1v, Ablang) for relative mutation effects.
PyRosetta's unique signal	Its lower correlation suggests it brings a complementary signal, possibly more focused on structural energetics rather than evolutionary or language-based reasoning.
Structure relaxation matters	While ML models are relatively indifferent to relaxation, PyRosetta benefits from it. If using PyRosetta, relax first.
VHH-specific caveat	Nanobodies are highly stable and compact—mutations can have subtle structural effects that PyRosetta may capture better than sequence-only models.
Use case synergy	You could use ML models for broad filtering and PyRosetta for fine-grained energetic validation (especially when expression/stability is critical).

Recommendations:
Use ML models (ESM1v, Ablang, Antifold) for high-throughput scoring
They provide consensus and are fast.

Use PyRosetta (with relaxation) as a final filter
Especially if you want to prioritize folding stability, interface binding, or core mutations.

Look at outliers where PyRosetta disagrees with ML models
These could be structurally important mutations (e.g. buried residues or H-bond networks) that are missed by ML models.

Consider integrating experimental validation
If you can validate a few mutants, you’ll know which model is most trustworthy for VHHs in your context.