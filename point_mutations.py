#!/usr/bin/env python3

import pandas as pd
import torch
import os
import subprocess
from ImmuneBuilder import ABodyBuilder2, NanoBodyBuilder2

AAs = "IVLCMAGTSWYPHEQDNKR"

try:
    from abnumber import Chain
    _NUMBERING_OK = True
    print("DEBUG: abnumber found, will use KABAT numbering")
except Exception:
    Chain = None
    _NUMBERING_OK = False
    print("WARNING: abnumber not available, fallback to simple positions")

# CONFIG
MODE = "mab"   # auto, vhh, mab
MODEL_TYPE = "antifold"  # ablang, esm2, esm1f, antiberta, antifold
INPUT_CSV = "/home/eva/0_point_mutation/results/TheraSAbDab_SeqStruc_OnlineDownload.csv"
OUTPUT = f"/home/eva/0_point_mutation/results/vhh_antifold_{MODEL_TYPE}.csv"
RUN_STRUCTURE_PREDICTION = True
PDB_OUTPUT_DIR = "/home/eva/0_point_mutation/pdbs"
ANTIFOLD_OUTPUT_DIR = "/home/eva/0_point_mutation/results/antifold"

os.makedirs(PDB_OUTPUT_DIR, exist_ok=True)
os.makedirs(ANTIFOLD_OUTPUT_DIR, exist_ok=True)

def load_model(format_type, model_type):
    if model_type == "ablang":
        import ablang2
        m = "ablang1-heavy" if format_type=="Nanobody" else "ablang2-paired"
        model = ablang2.pretrained(m, random_init=False, ncpu=1, device="cpu")
        model.freeze()
        return model
    elif model_type == "esm2":
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.eval()
        return (model, alphabet)
    elif model_type == "esm1f":
        import esm
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        model.eval()
        return (model, alphabet)
    elif model_type == "antiberta":
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        tokenizer = AutoTokenizer.from_pretrained("uclnlp/antiberta-base-uncased")
        model = AutoModelForMaskedLM.from_pretrained("uclnlp/antiberta-base-uncased")
        model.eval()
        return (model, tokenizer)
    elif model_type == "antifold":
        return None  # handled in worker script
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

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
                if wt not in tok_to_idx: continue
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
                if wt not in tok_to_idx: continue
                wt_ll = vh_logits[pos][tok_to_idx[wt]].item()
                for mt in AAs:
                    mut_ll = vh_logits[pos][tok_to_idx[mt]].item()
                    delta_ll = mut_ll - wt_ll
                    records.append(("VH",pos+1,wt,mt,delta_ll,mut_ll,wt_ll))
            for pos, wt in enumerate(vl_seq):
                if wt not in tok_to_idx: continue
                wt_ll = vl_logits[pos][tok_to_idx[wt]].item()
                for mt in AAs:
                    mut_ll = vl_logits[pos][tok_to_idx[mt]].item()
                    delta_ll = mut_ll - wt_ll
                    records.append(("VL",pos+1,wt,mt,delta_ll,mut_ll,wt_ll))
        return pd.DataFrame(records, columns=[
            "chain","pos","wt","mt","delta_log_likelihood","mut_log_likelihood","wt_log_likelihood"
        ])

    elif model_type in ["esm2","esm1f"]:
        esm_model, alphabet = model
        batch_converter = alphabet.get_batch_converter()
        records = []
        if vh_seq:
            data_vh = [("vh", vh_seq)]
            _, _, tokens = batch_converter(data_vh)
            with torch.no_grad():
                logits = esm_model(tokens)["logits"][0]
            for pos, wt in enumerate(vh_seq, start=1):
                wt_idx = alphabet.get_idx(wt)
                if wt_idx is None: continue
                wt_ll = logits[pos][wt_idx].item()
                for mt in AAs:
                    mt_idx = alphabet.get_idx(mt)
                    if mt_idx is None: continue
                    mut_ll = logits[pos][mt_idx].item()
                    delta_ll = mut_ll - wt_ll
                    records.append(("VH",pos,wt,mt,delta_ll,mut_ll,wt_ll))
        if vl_seq:
            data_vl = [("vl", vl_seq)]
            _, _, tokens = batch_converter(data_vl)
            with torch.no_grad():
                logits = esm_model(tokens)["logits"][0]
            for pos, wt in enumerate(vl_seq, start=1):
                wt_idx = alphabet.get_idx(wt)
                if wt_idx is None: continue
                wt_ll = logits[pos][wt_idx].item()
                for mt in AAs:
                    mt_idx = alphabet.get_idx(mt)
                    if mt_idx is None: continue
                    mut_ll = logits[pos][mt_idx].item()
                    delta_ll = mut_ll - wt_ll
                    records.append(("VL",pos,wt,mt,delta_ll,mut_ll,wt_ll))
        return pd.DataFrame(records, columns=[
            "chain","pos","wt","mt","delta_log_likelihood","mut_log_likelihood","wt_log_likelihood"
        ])

    elif model_type == "antifold":
        pdbfile = os.path.join(PDB_OUTPUT_DIR, f"{sample_name}.pdb")
        if not os.path.exists(pdbfile):
            run_abodybuilder2(vh_seq, vl_seq if format_type!="Nanobody" else None, pdbfile)

        print(f"Launching antifold worker subprocess for {sample_name}")
        subprocess.run(
            [
                "conda", "run", "-n", "antifold", "python", "antifold_worker.py",
                sample_name,
                PDB_OUTPUT_DIR,
                ANTIFOLD_OUTPUT_DIR,
                format_type
            ],
            check=True
        )

        import glob

        antifold_pattern = os.path.join(ANTIFOLD_OUTPUT_DIR, f"{sample_name}*.csv")
        antifold_files = glob.glob(antifold_pattern)

        if not antifold_files:
            raise FileNotFoundError(f"No antifold CSV found matching {antifold_pattern}")

        antifold_csv = antifold_files[0]
        print(f"Found antifold CSV: {antifold_csv}")

        df = pd.read_csv(antifold_csv)
        df["sample"] = sample_name
        return df


    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def run_abodybuilder2(vh_seq, vl_seq, output_path):
    if vl_seq:
        predictor = ABodyBuilder2()
        model = predictor.predict({'H': vh_seq, 'L': vl_seq})
    else:
        predictor = NanoBodyBuilder2()
        model = predictor.predict({'H': vh_seq})
    model.save(output_path)
    print(f"DEBUG: saved structure {output_path}")

def main():
    data = pd.read_csv(INPUT_CSV)

    if MODE == "vhh":
        data = data[
            data["Format"].str.lower().str.contains("nanobody|vhh", na=False)
        ].copy()
        print(f"Filtered data to {len(data)} nanobody entries in vhh mode.")

    if data.empty:
        print("No samples to process after filtering.")
        return

    print(f"\n=== Running {MODEL_TYPE} ===")
    per_model = []

    for sample_idx, (_, row) in enumerate(data.iterrows(), start=1):
        name = row["name"]
        vh = row["vh"].upper()
        vl = row["vl"].upper() if "vl" in row and pd.notna(row["vl"]) else ""
        format_str = str(row["Format"]).lower()

        if MODE=="auto":
            if "vhh" in format_str or "nanobody" in format_str:
                format_type = "Nanobody"
            elif "mab" in format_str:
                format_type = "VHVL"
            else:
                continue
        elif MODE=="vhh":
            format_type = "Nanobody"
        elif MODE=="mab":
            format_type = "VHVL"
        else:
            raise ValueError("bad MODE")

        print(f"  -> [{sample_idx}/{len(data)}] Scoring {name} with model {MODEL_TYPE}...")
        model = load_model(format_type, MODEL_TYPE)
        df = score_paired(vh, vl, model, format_type, MODEL_TYPE, sample_name=name)
        if MODEL_TYPE != "antifold":
            df = df.rename(columns={
                "delta_log_likelihood": f"delta_log_likelihood_{MODEL_TYPE}",
                "mut_log_likelihood": f"mut_log_likelihood_{MODEL_TYPE}",
                "wt_log_likelihood": f"wt_log_likelihood_{MODEL_TYPE}"
            })
        df["sample"] = name
        per_model.append(df)

    final = pd.concat(per_model, ignore_index=True)
    final.to_csv(OUTPUT, sep="\t", index=False)
    print(f"Done scoring {MODEL_TYPE}, results saved to {OUTPUT}")

if __name__ == "__main__":
    main()
