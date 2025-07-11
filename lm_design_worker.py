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


def validate_structure_loading(designer, pdb_path):
    print(f"\n=== Structure Validation ===")
    if designer._structure is None:
        print("ERROR: Structure is None!")
        return False

    print(f"Structure keys: {list(designer._structure.keys())}")

    if "seq" not in designer._structure:
        print("ERROR: No 'seq' field in structure!")
        return False

    if "coord" not in designer._structure:
        print("WARNING: No 'coord' field in structure - model may not have structural constraints!")

    seq = designer._structure["seq"]
    if isinstance(seq, tuple):
        seq = seq[0]

    print(f"Loaded sequence length: {len(seq)}")
    print(f"Sequence preview: {seq[:30]}...")

    return True


def get_valid_amino_acid_indices(designer):
    valid_indices = set()
    print("\n=== Valid Amino Acid Token Mapping ===")
    for aa in AMINO_ACIDS:
        try:
            idx = designer.alphabet.get_idx(aa)
            valid_indices.add(idx)
            print(f"{aa} → {idx}")
        except:
            print(f"WARNING: Could not get index for amino acid {aa}")
    print(f"Valid indices: {sorted(valid_indices)}")
    return valid_indices


def analyze_token_distribution(log_probs, valid_indices, position_sample=5):
    print(f"\n=== Token Distribution Analysis (first {position_sample} positions) ===")
    for pos in range(min(position_sample, log_probs.shape[1])):
        probs = torch.exp(log_probs[0, pos, :])
        top_k = 5
        top_probs, top_indices = torch.topk(probs, top_k)
        print(f"Position {pos}:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            is_valid = idx.item() in valid_indices
            print(f"  Top {i+1}: idx={idx.item()}, prob={prob.item():.4f}, valid={is_valid}")


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

        if not validate_structure_loading(designer, pdb_path):
            print("ERROR: Structure validation failed!")
            sys.exit(1)

        chain_id = "H"  # default for Nanobody

        valid_indices = get_valid_amino_acid_indices(designer)

        native_seq = designer._structure["seq"]
        if isinstance(native_seq, tuple):
            print("WARNING: seq was a tuple, unpacking first element.")
            native_seq = native_seq[0]

        native_seq = sanitize_sequence(native_seq)
        print("native_seq length:", len(native_seq), "sequence (first 20):", native_seq[:20])

        batch = designer._featurize()

        print("\n=== Batch Analysis ===")
        print(f"Batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: {type(value)}")

        log_probs = compute_log_likelihoods(batch, designer.model)

        analyze_token_distribution(log_probs, valid_indices)

        result_rows = []
        positive_delta_count = 0
        total_mutations = 0

        for i, wt_aa in enumerate(native_seq):
            wt_idx = designer.alphabet.get_idx(wt_aa)
            if wt_idx not in valid_indices:
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
                if mt_idx not in valid_indices:
                    continue

                try:
                    mt_log_prob = log_probs[0, i, mt_idx].item()
                except IndexError:
                    continue

                delta = mt_log_prob - wt_log_prob
                resi_label = str(i + 1)

                result_rows.append({
                    "chain": chain_id,
                    "pdb_residue": resi_label,
                    "wt": wt_aa,
                    "mt": mt_aa,
                    "mut_log_likelihood_lm_design": mt_log_prob,
                    "wt_log_likelihood_lm_design": wt_log_prob,
                    "delta_log_likelihood_lm_design": delta,
                    "sample": sample_name
                })

                total_mutations += 1
                if delta > 0:
                    positive_delta_count += 1

        print(f"\n=== Mutation Statistics ===")
        print(f"Total mutations: {total_mutations}")
        print(f"Positive delta mutations: {positive_delta_count}")
        print(f"Percentage positive: {positive_delta_count/total_mutations*100:.1f}%")

        if positive_delta_count / total_mutations > 0.4:
            print("WARNING: Unusually high percentage of positive delta mutations!")
            print("This suggests the model may not be properly constrained by the structure.")

        df = pd.DataFrame(result_rows)
        out_csv = os.path.join(LMDESIGN_OUTPUT_DIR, f"{sample_name}_point_mutation_scan.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved point mutation log-likelihood scan to: {out_csv}")

    except Exception as e:
        print(f"ERROR running LM-Design point mutation scan: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
