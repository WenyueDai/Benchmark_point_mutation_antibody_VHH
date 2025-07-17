#!/usr/bin/env python3

import os
import pandas as pd
import torch
import subprocess
import gc
import shutil

try:
    from antiberty import AntiBERTyRunner
    from abnumber import Chain
except ImportError:
    print("WARNING: AntiBERTy or abnumber not installed, some features may be unavailable.")
    
#TODO debug lm_design - need to have both sequence and structural loss (code can see from esm github)
    
# =============================
# CONFIGURATION
# =============================
MODE = "mab"   # auto (for all), vhh, mab
ORDER = "H,L"  # chain order for protein complex to consider for log likelihood calculation, 
MODEL_TYPES = ['ablang', 'esm1v', 'esm1f', 'antifold', 'pyrosetta']  # ablang, esm1v, esm1f, antiberta, antifold, nanobert, pyrosetta, lm_design (weird, dont use), tempro

INPUT_CSV = "/home/eva/0_point_mutation/playground_mAb_DMS/1MLC.csv"
PDB_OUTPUT_DIR = "/home/eva/0_point_mutation/pdbs"
ANTIFOLD_OUTPUT_DIR = "/home/eva/0_point_mutation/results/antifold"

os.makedirs(PDB_OUTPUT_DIR, exist_ok=True)
os.makedirs(ANTIFOLD_OUTPUT_DIR, exist_ok=True)

# =============================
# MODEL LOADERS
# =============================

def load_model(format_type, model_type):
    if model_type == "ablang":
        import ablang2
        m = "ablang1-heavy" if format_type.lower() == "nanobody" else "ablang2-paired"
        model = ablang2.pretrained(m, random_init=False, ncpu=1, device="cpu")
        model.freeze()
        return model
    return None

# =============================
# ABLANG SCORER
# =============================

def score_paired(vh_seq, vl_seq, model, format_type, model_type, sample_name=None):
    if format_type.lower() == "nanobody":
        fullseq = vh_seq
    else:
        fullseq = f"{vh_seq}|{vl_seq}"

    # Create a full sequence with a mask for mutation
    seqs = {}
    for i in range(len(fullseq)):
        # Skip the mask character if not in nanobody format or is a separator
        if format_type.lower() != "nanobody" and fullseq[i] == "|":
            continue
        newseq = fullseq[:i] + "*" + fullseq[i+1:]
        if format_type.lower() == "nanobody":
            chain, pos = "H", i + 1
        else:
            chain, pos = ("H", i + 1) if i < len(vh_seq) else ("L", i - len(vh_seq))
        seqs[(chain, pos)] = newseq

    records = []
    for (chain, pos), seq in seqs.items():
        idx = seq.index("*")
        call_params = model.tokenizer.__call__.__code__.co_varnames
        # for ablang1-heavy (used in nanobody), we need to pass `w_extra_tkns=False` because this arg dont exist
        tokenized = model.tokenizer([seq], pad=True, w_extra_tkns=False, device="cpu") if "w_extra_tkns" in call_params else model.tokenizer([seq], pad=True, device="cpu")
        # Turn off autograd to save memory (no gradients needed for inference)
        with torch.no_grad():
            # Get logits for the sequence with the mutation mask
            logits = model.AbLang(tokenized)[0]
        # Set logits for special tokens to a very low value to avoid them being selected
        special_tokens = getattr(model.tokenizer, "all_special_tokens", None)
        if special_tokens:
            logits[:, special_tokens] = -float("inf")

        try:
            # Get the token ID for the wildtype amino acid at the mutation position
            wt_token = model.tokenizer.aa_to_token[fullseq[idx]]
        except KeyError:
            print(f"WARNING: wildtype token {fullseq[idx]} not found in tokenizer; skipping")
            continue

        wt_ll = logits[idx, wt_token].item()
        # Only use standard 20 amino acids for mutation
        VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")
        aa_list = [aa for aa in model.tokenizer.aa_to_token.keys() if aa in VALID_AAS]

        for aa in aa_list:
            if aa == fullseq[idx]:
                continue
            aa_token = model.tokenizer.aa_to_token[aa]
            mut_ll = logits[idx, aa_token].item()
            delta_ll = mut_ll - wt_ll
            records.append({
                "sample": sample_name,
                "chain": chain,
                "pos": pos,
                "wt": fullseq[idx],
                "mt": aa,
                "wt_log_likelihood_ablang2": wt_ll,
                "mut_log_likelihood_ablang2": mut_ll,
                "delta_log_likelihood_ablang2": delta_ll,
            })

    return pd.DataFrame.from_records(records)

# =============================
# ABodyBuilder
# =============================

def run_abodybuilder2(vh_seq, vl_seq, output_path):
    from ImmuneBuilder import ABodyBuilder2, NanoBodyBuilder2
    predictor = ABodyBuilder2() if vl_seq else NanoBodyBuilder2()
    model = predictor.predict({'H': vh_seq, 'L': vl_seq} if vl_seq else {'H': vh_seq})
    model.save(output_path)
    print(f"Saved structure to {output_path}")
    # Perform sequential renumbering starting from 1
    renumber_pdb_sequential(output_path)

    # Call relax using conda environment
    relaxed_script = os.path.abspath("worker_py/pyrosetta_relax.py")
    subprocess.run([
        "conda", "run", "-n", "pyrosetta", "python", relaxed_script, output_path
    ], check=True)

def renumber_pdb_sequential(pdb_path, output_path=None):
    """
    Rewrites a PDB file with sequential residue numbering for all chains.
    """
    if output_path is None:
        output_path = pdb_path

    new_lines = []
    current_resi = {}
    resi_map = {}

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                new_lines.append(line)
                continue

            chain = line[21]
            resnum = int(line[22:26])
            icode = line[26]

            key = (chain, resnum, icode)
            if chain not in current_resi:
                current_resi[chain] = 1
                resi_map[chain] = {}

            if key not in resi_map[chain]:
                resi_map[chain][key] = current_resi[chain]
                current_resi[chain] += 1

            new_resnum = resi_map[chain][key]
            newline = line[:22] + f"{new_resnum:>4}" + line[26:]
            new_lines.append(newline)

    with open(output_path, "w") as f:
        f.writelines(new_lines)
    print(f"Renumbered PDB written to {output_path}")

# =============================
# ANTI-BERTY MUT SCAN
# =============================

def mutation_scan_paired(antiberty, vh_seq, vl_seq=None, batch_size=16):
    AAs = "ACDEFGHIKLMNPQRSTVWY"
    records, wt_seqs, chains = [], [vh_seq], ["H"]
    if vl_seq:
        wt_seqs.append(vl_seq)
        chains.append("L")
    wt_plls = antiberty.pseudo_log_likelihood(wt_seqs, batch_size=1)
    wt_pll_dict = dict(zip(chains, [pll.item() for pll in wt_plls]))

    for chain_label, seq in zip(chains, wt_seqs):
        mutants, info = [], []
        for pos, wt in enumerate(seq):
            for mt in AAs:
                mutated = seq[:pos] + mt + seq[pos+1:]
                mutants.append([mutated] if chain_label == "H" and not vl_seq else [vh_seq if chain_label == "L" else mutated, vl_seq if chain_label == "H" else mutated])
                info.append((chain_label, pos + 1, wt, mt))
        flat = [item for pair in mutants for item in pair]
        # Ensure input is on the same device as the model
        # flat: list of sequences (str)
        # antiberty.tokenizer.encode(...): converts each sequence to a list of token IDs
        # torch.tensor(...).to(device): puts each tokenized tensor on the same device as AntiBERTy model
        # Pass the list of tensors (flat_encoded) to the model.
        mut_plls = antiberty.pseudo_log_likelihood(flat, batch_size=batch_size)

        for idx, (chain, pos, wt, mt) in enumerate(info):
            records.append({
                "chain": chain,
                "pos": pos,
                "wt": wt,
                "mt": mt,
                "mt_log_likelihood_antiberty": mut_plls[idx].item(),
                "wt_log_likelihood_antiberty": wt_pll_dict[chain],
                "delta_log_likelihood_antiberty": mut_plls[idx].item() - wt_pll_dict[chain]
            })

    return pd.DataFrame(records)

def ensure_pdb_exists(name, vh, vl, format_type):
    """
    Ensure that the PDB file exists for the given sample.
    If not, generate it using ABodyBuilder2.
    """
    pdbfile = os.path.join(PDB_OUTPUT_DIR, f"{name}.pdb")
    if not os.path.exists(pdbfile):
        print(f"Generating PDB for {name} to {pdbfile}")
        vl_clean = None if vl in ["", "NA", "na", None] else vl
        run_abodybuilder2(vh, vl_clean if format_type == "VHVL" else None, pdbfile)
    else:
        renumber_pdb_sequential(pdbfile)
        print(f"PDB file already exists for {name}, skipping ABodyBuilder, but renumbering.")
    return pdbfile

def run_worker_script(name, vh, vl, format_type, mutate_str, script_path, env, extra_args=None):
    """
    Run the worker script for the given sample.
    """
    args = [
        "conda", "run", "-n", env, "python", script_path,
        name, vh, vl if format_type == "VHVL" else "NA",
        "--mutate", mutate_str, "--format", format_type
    ]
    if extra_args:
        args.extend(extra_args)
    print(f"Launching: {' '.join(args)}")
    subprocess.run(args, check=True)
    
def handle_ablang(vh, vl, model, format_type, name, output):
    df = score_paired(vh, vl, model, format_type, "ablang", sample_name=name)
    df["sample"] = name
    df.to_csv(output, sep="\t", mode="a", header=not os.path.exists(output), index=False)
    print(f"Results for {name} written to {output}")
    
def handle_antiberta(vh, vl, format_type, name, output, runner):
    seqs = [vh] if format_type == "Nanobody" else [vh, vl]
    if format_type == "VHVL" and (not vl or vl == "NA"):
        print(f"Skipping {name}: missing VL sequence")
        return
    device = runner.device
    pll_scores = runner.pseudo_log_likelihood(seqs, batch_size=1)
    mut_df = mutation_scan_paired(runner, vh, vl if vl else None, batch_size=16)
    mut_df["sample"] = name
    mut_df.to_csv(output, sep="\t", mode="a", header=not os.path.exists(output), index=False)
    print(f"Results for {name} written to {output}")
    
def get_format_type(format_str, mode):
    format_str = str(format_str).lower()
    if mode == "auto":
        if "nanobody" in format_str or "vhh" in format_str:
            return "Nanobody"
        elif "mab" in format_str:
            return "VHVL"
        else:
            return None
    elif mode == "vhh":
        return "Nanobody"
    elif mode == "mab":
        return "VHVL"
    else:
        raise ValueError("Invalid MODE")

# =============================
# MAIN
# =============================

def main():
    for MODEL_TYPE in MODEL_TYPES:
        print(f"\n=== Running MODEL_TYPE: {MODEL_TYPE} ===")
        OUTPUT = f"/home/eva/0_point_mutation/results/{MODEL_TYPE}/{MODE}_{MODEL_TYPE}.csv"
        os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
        
        ablang_model = None
        antiberty = None
        
        if MODEL_TYPE == "ablang":
            ablang_model = load_model(MODE, MODEL_TYPE)
            # Compatibility patch
            tokenizer = ablang_model.tokenizer
            print(f"Tokenizer keys: {dir(ablang_model.tokenizer)}")
            # ablang1-heavy and ablang2 have different tokenizer attributes
            if not hasattr(tokenizer, "aa_to_token"):
                if hasattr(tokenizer, "vocab_to_token"):
                    tokenizer.aa_to_token = tokenizer.vocab_to_token
                else:
                    raise AttributeError("Tokenizer lacks both 'aa_to_token' and 'vocab_to_token'")
            if not hasattr(tokenizer, "vocab_to_token"):
                if hasattr(tokenizer, "aa_to_token"):
                    tokenizer.vocab_to_token = tokenizer.aa_to_token
                else:
                    raise AttributeError("Tokenizer lacks both 'aa_to_token' and 'vocab_to_token'")
                
        elif MODEL_TYPE == "antiberta":
            antiberty = AntiBERTyRunner()
            antiberty.model = antiberty.model.to("cpu")
            antiberty.device = "cpu"
            print(f"Loaded AntiBERTy on {antiberty.device}")
            
        for sample_idx, (_, row) in enumerate(pd.read_csv(INPUT_CSV).iterrows(), start=1):
            name = row["name"]
            vh = row["vh"].upper()
            vl = row["vl"].upper() if "vl" in row and pd.notna(row["vl"]) else ""
            format_str = str(row["Format"]).lower()

            format_type = get_format_type(format_str, MODE)
            if not format_type:
                continue

            print(f"[{sample_idx}/{len(data)}] Processing {name} ({format_type})...")

            try:
                if MODEL_TYPE == "antiberta":
                    handle_antiberta(vh, vl, format_type, name, OUTPUT, antiberty)
                elif MODEL_TYPE == "ablang":
                    handle_ablang(vh, vl, ablang_model, format_type, name, OUTPUT)
                elif MODEL_TYPE in ["esm1v", "nanobert", "lm_design", "tempro"]:
                    ensure_pdb_exists(name, vh, vl, format_type)
                    script_map = {
                        "nanobert": ("worker_py/nanobert_worker.py", "antiberty"),
                        "esm1v": ("worker_py/esm1v_worker.py", "esm"),
                        "lm_design": ("worker_py/lm_design_worker.py", "lm_design"),
                        "tempro": ("worker_py/thermo_worker.py", "esm")
                    }
                    script_path, env = script_map[MODEL_TYPE]
                    run_worker_script(name, vh, vl, format_type, "H" if format_type == "Nanobody" else "H,L", script_path, env)
                    
                elif MODEL_TYPE in ["antifold", "esm1f", "pyrosetta"]:
                    pdbfile = ensure_pdb_exists(name, vh, vl, format_type)
                    script_map = {
                        "antifold": ("worker_py/antifold_worker.py", "antifold"),
                        "esm1f": ("worker_py/esm1f_worker.py", "esm1f"),
                        "pyrosetta": ("worker_py/pyrosetta_worker.py", "pyrosetta")
                    }
                    script_path, env = script_map[MODEL_TYPE]
                    vl_clean = vl if format_type == "VHVL" and vl not in ["", "NA", "na", None] else "NA"
                    mutate_str = "H" if format_type == "Nanobody" else "H,L
                    extra_args = ["--order", ORDER, "--nogpu"] if MODEL_TYPE in ["esm1f", "pyrosetta"] else []
                    run_worker_script(name, vh, vl_clean, format_type, mutate_str, script_path, env, extra_args)
            
            except subprocess.CalledProcessError as e:
                print(f"Worker script failed for {name}: {e}")
                continue
            except Exception as e:
                print(f"Failed on {name}: {e}")
                continue
            finally:
                # Clear memory and cache
                print("Cleaning up after model run...")
                gc.collect()
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
