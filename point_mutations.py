#!/usr/bin/env python3

import os
import pandas as pd
import torch
import subprocess

try:
    from antiberty import AntiBERTyRunner
except ImportError:
    AntiBERTyRunner = None

try:
    from abnumber import Chain
    _NUMBERING_OK = True
    print("DEBUG: abnumber found, will use KABAT numbering")
except Exception:
    Chain = None
    _NUMBERING_OK = False
    print("WARNING: abnumber not available, fallback to simple positions")

# =============================
# CONFIGURATION
# =============================
MODE = "vhh"   # auto (for all), vhh, mab
MODEL_TYPE = "pyrosetta"  # ablang, esm1v, esm1f, antiberta, antifold, nanobert, pyrosetta, lm_design (weird, dont use), tempro

INPUT_CSV = "/home/eva/0_point_mutation/results/TheraSAbDab_SeqStruc_OnlineDownload.csv"
OUTPUT = f"/home/eva/0_point_mutation/results/{MODEL_TYPE}/{MODE}_{MODEL_TYPE}.csv"
PDB_OUTPUT_DIR = "/home/eva/0_point_mutation/pdbs"
ANTIFOLD_OUTPUT_DIR = "/home/eva/0_point_mutation/results/antifold"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
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

    seqs = {}
    for i in range(len(fullseq)):
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
        tokenized = model.tokenizer([seq], pad=True, w_extra_tkns=False, device="cpu") if "w_extra_tkns" in call_params else model.tokenizer([seq], pad=True, device="cpu")
        with torch.no_grad():
            logits = model.AbLang(tokenized)[0]

        special_tokens = getattr(model.tokenizer, "all_special_tokens", None)
        if special_tokens:
            logits[:, special_tokens] = -float("inf")

        try:
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
                "wt_log_likelihood": wt_ll,
                "mut_log_likelihood": mut_ll,
                "delta_log_likelihood": delta_ll,
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
    # Perform renumbering
    renumber_pdb_sequential(output_path)

    # Call relax using conda environment
    relaxed_script = os.path.abspath("pyrosetta_relax.py")
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
                "pll_mutant": mut_plls[idx].item(),
                "pll_wildtype": wt_pll_dict[chain],
                "delta_pll": mut_plls[idx].item() - wt_pll_dict[chain]
            })

    return pd.DataFrame(records)

# =============================
# MAIN
# =============================

def main():
    data = pd.read_csv(INPUT_CSV)
    if MODE == "vhh":
        data = data[data["Format"].str.lower().str.contains("nanobody|vhh", na=False)]
        print(f"Filtered to {len(data)} entries in vhh mode.")
    if data.empty:
        print("No samples to process.")
        return

    print(f"\n=== Running MODEL_TYPE: {MODEL_TYPE} ===")

    ablang_model = None
    if MODEL_TYPE == "ablang":
        ablang_model = load_model(MODE, MODEL_TYPE)
        # Compatibility patch
        tokenizer = ablang_model.tokenizer
        print(f"Tokenizer keys: {dir(ablang_model.tokenizer)}")
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


    if MODEL_TYPE == "antiberta":
        if AntiBERTyRunner is None:
            print("ERROR: AntiBERTy not installed.")
            return
        antiberty = AntiBERTyRunner()
        antiberty.model = antiberty.model.to("cpu")
        antiberty.device = "cpu"
        print(f"Loaded AntiBERTy on {antiberty.device}")

    for sample_idx, (_, row) in enumerate(data.iterrows(), start=1):
        name = row["name"]
        vh = row["vh"].upper()
        vl = row["vl"].upper() if "vl" in row and pd.notna(row["vl"]) else ""
        format_str = str(row["Format"]).lower()

        if MODE == "auto":
            if "nanobody" in format_str or "vhh" in format_str:
                format_type = "Nanobody"
            elif "mab" in format_str:
                format_type = "VHVL"
            else:
                continue
        elif MODE == "vhh":
            format_type = "Nanobody"
        elif MODE == "mab":
            format_type = "VHVL"
        else:
            raise ValueError("Invalid MODE")

        print(f"[{sample_idx}/{len(data)}] Processing {name} ({format_type})...")

        try:
            if MODEL_TYPE == "antiberta":
                seqs = [vh] if format_type == "Nanobody" else [vh, vl]
                if format_type == "VHVL" and (not vl or vl == "NA"):
                    print(f"Skipping {name}: missing VL sequence")
                    continue
                device = antiberty.device
                pll_scores = antiberty.pseudo_log_likelihood(seqs, batch_size=1)
                mut_df = mutation_scan_paired(antiberty, vh, vl if vl else None, batch_size=16)
                mut_df = mut_df.rename(columns={
                    "pll_mutant": f"mut_log_likelihood_{MODEL_TYPE}",
                    "pll_wildtype": f"wt_log_likelihood_{MODEL_TYPE}",
                    "delta_pll": f"delta_log_likelihood_{MODEL_TYPE}"
                })
                mut_df["sample"] = name
                mut_df.to_csv(OUTPUT, sep="\t", mode="a", header=not os.path.exists(OUTPUT), index=False)
                print(f"Results for {name} written to {OUTPUT}")

            elif MODEL_TYPE in ["antifold", "esm1v", "esm1f", "nanobert", "pyrosetta", "lm_design", "tempro"]:
                pdbfile = os.path.join(PDB_OUTPUT_DIR, f"{name}.pdb")
                if MODEL_TYPE in ["antifold", "pyrosetta", "lm_design", "esm1f"]:
                    if not os.path.exists(pdbfile):
                        print(f"Generating PDB for {name} → {pdbfile}")
                        vl_clean = None if vl in ["", "NA", "na", None] else vl
                        run_abodybuilder2(vh, vl_clean if format_type == "VHVL" else None, pdbfile)
                    else:
                        print(f"PDB file already exists for {name}, skipping ABodyBuilder.")

                script_map = {
                    "antifold": ("antifold_worker.py", "antifold"),
                    "pyrosetta": ("pyrosetta_worker.py", "pyrosetta"),
                    "nanobert": ("nanobert_worker.py", "antiberty"),
                    "esm1v": ("esm1v_worker.py", "esm"),
                    "esm1f": ("esm1f_worker.py", "esm1f"),
                    "lm_design": ("lm_design_worker.py", "lm_design"),
                    "tempro": ("thermo_worker.py", "esm")
                }
                worker_script, env = script_map[MODEL_TYPE]
                worker_args = [
                    "conda", "run", "-n", env, "python", worker_script,
                    name, vh, vl if format_type == "VHVL" else "NA", format_type
                ]

                print(f"Launching: {' '.join(worker_args)}")
                subprocess.run(worker_args, check=True)

            elif MODEL_TYPE == "ablang":
                df = score_paired(vh, vl, ablang_model, format_type, MODEL_TYPE, sample_name=name)
                df = df.rename(columns={
                    "delta_log_likelihood": f"delta_log_likelihood_{MODEL_TYPE}",
                    "mut_log_likelihood": f"mut_log_likelihood_{MODEL_TYPE}",
                    "wt_log_likelihood": f"wt_log_likelihood_{MODEL_TYPE}"
                })
                df["sample"] = name
                df = df[[
                    "chain", "pos", "wt", "mt",
                    f"mut_log_likelihood_{MODEL_TYPE}",
                    f"wt_log_likelihood_{MODEL_TYPE}",
                    f"delta_log_likelihood_{MODEL_TYPE}",
                    "sample"
                ]]
                df.to_csv(OUTPUT, sep="\t", mode="a", header=not os.path.exists(OUTPUT), index=False)
                print(f"Results for {name} written to {OUTPUT}")

        except subprocess.CalledProcessError as e:
            print(f"Worker script failed for {name}: {e}")
            continue
        except Exception as e:
            print(f"Failed on {name}: {e}")
            continue

if __name__ == "__main__":
    main()
