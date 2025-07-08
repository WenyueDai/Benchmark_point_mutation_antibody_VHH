#!/usr/bin/env python3

import sys
import os
import torch
import pandas as pd
from byprot.utils.config import compose_config as Cfg
from byprot.tasks.fixedbb.designer import Designer

PDB_INPUT_DIR = "/home/eva/0_point_mutation/pdbs"
LMDESIGN_OUTPUT_DIR = "/home/eva/0_point_mutation/results/lmdesign"
LM_MODEL_PATH = "/home/eva/0_point_mutation/ByProt-main/checkpoint/lm_design_esm2_650m"

os.makedirs(LMDESIGN_OUTPUT_DIR, exist_ok=True)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def compute_log_likelihoods(batch, model, cuda=True):
    with torch.no_grad():
        if cuda:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        output = model(batch)
        if isinstance(output, tuple):
            output = output[0]

        if isinstance(output, dict) and "logits" in output:
            logits = output["logits"]
        elif torch.is_tensor(output):
            logits = output
        else:
            raise TypeError(f"Unexpected model output type: {type(output)}")

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        print("DEBUG log_probs shape:", log_probs.shape)
        return log_probs.cpu()

def sanitize_sequence(seq):
    return ''.join(res for res in seq if res in AMINO_ACIDS)

def main():
    if len(sys.argv) != 5:
        print("Usage: python lm_design_worker.py <sample_name> <vh_seq> <vl_seq_or_NA> <format_type>")
        sys.exit(1)

    sample_name = sys.argv[1]
    format_type = sys.argv[4]
    pdb_path = os.path.join(PDB_INPUT_DIR, f"{sample_name}.pdb")

    if not os.path.exists(pdb_path):
        print(f"ERROR: PDB file does not exist: {pdb_path}")
        sys.exit(1)

    try:
        cfg = Cfg(
            cuda=True,
            generator=Cfg(
                max_iter=3,
                strategy="denoise",
                temperature=0,
                eval_sc=False,
            )
        )

        designer = Designer(experiment_path=LM_MODEL_PATH, cfg=cfg)

        print(f"Loading PDB structure from: {pdb_path}")
        designer.set_structure(pdb_path)
        
        print("\n=== Full Token Index Mapping ===")
        max_index = 50  # Adjust as needed
        for idx in range(max_index):
            try:
                tok = designer.alphabet.get_tok(idx)
                print(f"Index {idx:2d}: {repr(tok)}")
            except:
                break
            
        print("\n=== Amino Acid to Index Mapping ===")
        for aa in AMINO_ACIDS:
            idx = designer.alphabet.get_idx(aa)
            print(f"{aa} → {idx}")


        if designer._structure is None or "seq" not in designer._structure:
            raise ValueError("Structure not properly loaded or missing 'seq' field.")

        native_seq = designer._structure["seq"]
        if isinstance(native_seq, tuple):
            print("WARNING: seq was a tuple, unpacking first element.")
            native_seq = native_seq[0]

        native_seq = sanitize_sequence(native_seq)
        print("native_seq length:", len(native_seq), "sequence (first 20):", native_seq[:20])

        chain_id = "H"  # default for Nanobody
        batch = designer._featurize()
        log_probs = compute_log_likelihoods(batch, designer.model)

        # Limit to valid amino acid indices (L to C = index 4–23)
        VALID_IDX = set(range(4, 24))

        result_rows = []
        for i, wt_aa in enumerate(native_seq):
            wt_idx = designer.alphabet.get_idx(wt_aa)
            if wt_idx not in VALID_IDX:
                print(f"Skipping wt residue {wt_aa} at pos {i+1} → idx {wt_idx}")
                continue

            try:
                wt_log_prob = log_probs[0, i, wt_idx].item()
            except IndexError:
                continue

            for mt_aa in AMINO_ACIDS:
                if mt_aa == wt_aa:
                    continue
                mt_idx = designer.alphabet.get_idx(mt_aa)
                if mt_idx not in VALID_IDX:
                    continue

                try:
                    mt_log_prob = log_probs[0, i, mt_idx].item()
                except IndexError:
                    continue

                delta = mt_log_prob - wt_log_prob
                result_rows.append({
                    "chain": chain_id,
                    "pos": i + 1,
                    "wt": wt_aa,
                    "mt": mt_aa,
                    "mut_log_likelihood_lm_design": mt_log_prob,
                    "wt_log_likelihood_lm_design": wt_log_prob,
                    "delta_log_likelihood_lm_design": delta,
                    "sample": sample_name
                })

        df = pd.DataFrame(result_rows)
        out_csv = os.path.join(LMDESIGN_OUTPUT_DIR, f"{sample_name}_point_mutation_scan.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved point mutation log-likelihood scan to: {out_csv}")

    except Exception as e:
        print(f"ERROR running LM-Design point mutation scan: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
