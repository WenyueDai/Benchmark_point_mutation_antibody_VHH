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

AAs = "IVLCMAGTSWYPHEQDNKR"

# CONFIG
MODE = "vhh"   # auto, vhh, mab
MODEL_TYPE = "antiberta"  # ablang, esm2, esm1f, antiberta, antifold
INPUT_CSV = "/home/eva/0_point_mutation/results/TheraSAbDab_SeqStruc_OnlineDownload.csv"
OUTPUT = f"/home/eva/0_point_mutation/results/{MODEL_TYPE}/{MODE}_{MODEL_TYPE}.csv"
PDB_OUTPUT_DIR = "/home/eva/0_point_mutation/pdbs"
ANTIFOLD_OUTPUT_DIR = "/home/eva/0_point_mutation/results/antifold"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
os.makedirs(PDB_OUTPUT_DIR, exist_ok=True)
os.makedirs(ANTIFOLD_OUTPUT_DIR, exist_ok=True)

def load_model(format_type, model_type):
    if model_type == "ablang":
        import ablang2
        m = "ablang1-heavy" if format_type == "Nanobody" else "ablang2-paired"
        model = ablang2.pretrained(m, random_init=False, ncpu=1, device="cpu")
        model.freeze()
        return model
    else:
        return None  # handled below

def score_paired(vh_seq, vl_seq, model, format_type, model_type, sample_name=None):
    if model_type == "ablang":
        if format_type == "Nanobody":
            seqs = [vh_seq]
            tokenized = model.tokenizer(seqs, pad=True, device="cpu")
            with torch.no_grad():
                logits = model.AbLang(tokenized)[0]
            logits = logits[1:len(vh_seq)+1]
            records = []
            tok_to_idx = {aa: idx for idx, aa in enumerate(AAs)}
            for pos, wt in enumerate(vh_seq):
                if wt not in tok_to_idx:
                    continue
                wt_ll = logits[pos][tok_to_idx[wt]].item()
                for mt in AAs:
                    mut_ll = logits[pos][tok_to_idx[mt]].item()
                    delta_ll = mut_ll - wt_ll
                    records.append(("VH", pos+1, wt, mt, delta_ll, mut_ll, wt_ll))
        else:
            pair = f"{vh_seq}|{vl_seq}"
            tokenized = model.tokenizer([pair], pad=True, w_extra_tkns=False, device="cpu")
            with torch.no_grad():
                logits = model.AbLang(tokenized)[0]
            vh_len = len(vh_seq)
            vl_len = len(vl_seq)
            vh_logits = logits[1:vh_len+1]
            vl_logits = logits[vh_len+2:vh_len+2+vl_len]
            records = []
            tok_to_idx = {aa: idx for idx, aa in enumerate(AAs)}
            for pos, wt in enumerate(vh_seq):
                if wt not in tok_to_idx:
                    continue
                wt_ll = vh_logits[pos][tok_to_idx[wt]].item()
                for mt in AAs:
                    mut_ll = vh_logits[pos][tok_to_idx[mt]].item()
                    delta_ll = mut_ll - wt_ll
                    records.append(("VH", pos+1, wt, mt, delta_ll, mut_ll, wt_ll))
            for pos, wt in enumerate(vl_seq):
                if wt not in tok_to_idx:
                    continue
                wt_ll = vl_logits[pos][tok_to_idx[wt]].item()
                for mt in AAs:
                    mut_ll = vl_logits[pos][tok_to_idx[mt]].item()
                    delta_ll = mut_ll - wt_ll
                    records.append(("VL", pos+1, wt, mt, delta_ll, mut_ll, wt_ll))
        return pd.DataFrame(records, columns=[
            "chain", "pos", "wt", "mt",
            "delta_log_likelihood", "mut_log_likelihood", "wt_log_likelihood"
        ])
    else:
        raise NotImplementedError(
            f"Direct scoring for {model_type} is not supported in this script — handled below."
        )

def run_abodybuilder2(vh_seq, vl_seq, output_path):
    from ImmuneBuilder import ABodyBuilder2, NanoBodyBuilder2
    if vl_seq:
        predictor = ABodyBuilder2()
        model = predictor.predict({'H': vh_seq, 'L': vl_seq})
    else:
        predictor = NanoBodyBuilder2()
        model = predictor.predict({'H': vh_seq})
    model.save(output_path)
    print(f"Saved structure to {output_path}")

def mutation_scan_paired(antiberty, vh_seq, vl_seq=None, batch_size=16):
    """
    For a given VH (and optional VL), systematically mutate each residue
    to each of the other 19 amino acids and calculate PLL, in *batches*.
    Returns a dataframe with chain/pos/wt/mt/pll.
    """
    AAs = "IVLCMAGTSWYPHEQDNKR"
    records = []

    # VH mutations
    vh_mutants = []
    vh_info = []
    for pos, wt in enumerate(vh_seq):
        for mt in AAs:
            if mt == wt:
                continue
            mutated_vh = vh_seq[:pos] + mt + vh_seq[pos+1:]
            if vl_seq:
                vh_mutants.append([mutated_vh, vl_seq])
            else:
                vh_mutants.append([mutated_vh])
            vh_info.append( ("H", pos+1, wt, mt) )

    # flatten VH pairs
    vh_flat = []
    for pair in vh_mutants:
        vh_flat.extend(pair)
    vh_plls = antiberty.pseudo_log_likelihood(vh_flat, batch_size=batch_size)
    for idx, info in enumerate(vh_info):
        records.append({
            "chain": info[0],
            "pos": info[1],
            "wt": info[2],
            "mt": info[3],
            "pll": vh_plls[idx].item()
        })

    # VL mutations if present
    if vl_seq:
        vl_mutants = []
        vl_info = []
        for pos, wt in enumerate(vl_seq):
            for mt in AAs:
                if mt == wt:
                    continue
                mutated_vl = vl_seq[:pos] + mt + vl_seq[pos+1:]
                vl_mutants.append([vh_seq, mutated_vl])
                vl_info.append( ("L", pos+1, wt, mt) )

        vl_flat = []
        for pair in vl_mutants:
            vl_flat.extend(pair)
        vl_plls = antiberty.pseudo_log_likelihood(vl_flat, batch_size=batch_size)
        for idx, info in enumerate(vl_info):
            records.append({
                "chain": info[0],
                "pos": info[1],
                "wt": info[2],
                "mt": info[3],
                "pll": vl_plls[idx].item()
            })

    return pd.DataFrame(records)


def main():
    data = pd.read_csv(INPUT_CSV)

    if MODE == "vhh":
        data = data[data["Format"].str.lower().str.contains("nanobody|vhh", na=False)].copy()
        print(f"Filtered data to {len(data)} nanobody entries in vhh mode.")

    if data.empty:
        print("No samples to process after filtering.")
        return

    print(f"\n=== Running {MODEL_TYPE} ===")

    if MODEL_TYPE == "antiberta":
        if AntiBERTyRunner is None:
            print("ERROR: AntiBERTy is not installed.")
            return
        antiberty = AntiBERTyRunner()
        print(f"Loaded AntiBERTy on {antiberty.device}")

    ablang_model = None

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
            raise ValueError("bad MODE")

        print(f"  -> [{sample_idx}/{len(data)}] Processing {name}...")

        if MODEL_TYPE == "antiberta":
            try:
                if format_type == "Nanobody":
                    seqs = [vh]
                    chains = ["H"]
                else:
                    if not vl or vl == "NA":
                        print(f"Skipping {name}: missing VL for VHVL mode")
                        continue
                    seqs = [vh, vl]
                    chains = ["H", "L"]

                pll_scores = antiberty.pseudo_log_likelihood(
                    seqs,
                    batch_size=1
                )
                
                # run mutation scan for both nanobody or mab
                print(f"Running mutation scan on {name}...")
                mut_df = mutation_scan_paired(antiberty, vh, vl if vl else None, batch_size=16)
                mut_df["sample"] = name
                mut_csv = f"/home/eva/0_point_mutation/results/{MODEL_TYPE}/{name}_mutation_scan.csv"
                mut_df.to_csv(mut_csv, index=False)
                print(f"Mutation scan saved to {mut_csv}")


                records = []
                for chain_id, seq, pll in zip(chains, seqs, pll_scores):
                    records.append((chain_id, len(seq), pll.item(), name))

                tidy_df = pd.DataFrame(records, columns=[
                    "chain", "sequence_length",
                    "pseudo_log_likelihood_antiberty",
                    "sample"
                ])

                if not os.path.exists(OUTPUT):
                    tidy_df.to_csv(OUTPUT, sep="\t", index=False)
                else:
                    tidy_df.to_csv(OUTPUT, sep="\t", mode="a", header=False, index=False)

                print(f"Wrote results for {name} to {OUTPUT}")

            except Exception as e:
                print(f"Failed on {name}: {e}")
                continue

        elif MODEL_TYPE in ["antifold", "esm2", "esm1f"]:
            if MODEL_TYPE == "antifold":
                pdbfile = os.path.join(PDB_OUTPUT_DIR, f"{name}.pdb")
                if not os.path.exists(pdbfile):
                    run_abodybuilder2(vh, vl if format_type != "Nanobody" else None, pdbfile)
                worker_script = "antifold_worker.py"
                env_name = "antifold"
                worker_args = [
                    "conda", "run", "-n", env_name, "python", worker_script,
                    name, PDB_OUTPUT_DIR, ANTIFOLD_OUTPUT_DIR, format_type
                ]
            else:
                worker_script = f"{MODEL_TYPE}_worker.py"
                env_name = "esm"
                worker_args = [
                    "conda", "run", "-n", env_name, "python", worker_script,
                    name, vh, vl, format_type
                ]

            print(f"Launching worker subprocess: {' '.join(worker_args)}")
            try:
                subprocess.run(worker_args, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Worker script failed for {name}: {e}")
                continue

        elif MODEL_TYPE == "ablang":
            if ablang_model is None:
                ablang_model = load_model(format_type, MODEL_TYPE)
            try:
                df = score_paired(vh, vl, ablang_model, format_type, MODEL_TYPE, sample_name=name)
                df = df.rename(columns={
                    "delta_log_likelihood": f"delta_log_likelihood_{MODEL_TYPE}",
                    "mut_log_likelihood": f"mut_log_likelihood_{MODEL_TYPE}",
                    "wt_log_likelihood": f"wt_log_likelihood_{MODEL_TYPE}"
                })
                df["sample"] = name
                if not os.path.exists(OUTPUT):
                    df.to_csv(OUTPUT, sep="\t", index=False)
                else:
                    df.to_csv(OUTPUT, sep="\t", mode="a", header=False, index=False)
                print(f"Wrote results for {name} to {OUTPUT}")
            except Exception as e:
                print(f"Failed scoring with ablang for {name}: {e}")
                continue

        else:
            print(f"Unknown MODEL_TYPE {MODEL_TYPE}")
            continue

    print("All done.")

if __name__ == "__main__":
    main()
