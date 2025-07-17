#!/usr/bin/env python3

import sys
import os
import pandas as pd
from transformers import pipeline, RobertaTokenizer, AutoModelForMaskedLM

def scan_nanobert(sequence, chain_id, sample_name, top_k=20):
    """
    For a given sequence, systematically mask each position and get top_k predictions
    from nanoBERT. Also computes wt_probability, mut_probability, and delta_probability.
    """
    results = []

    # load nanoBERT locally and offline
    tokenizer = RobertaTokenizer.from_pretrained("NaturalAntibody/nanoBERT", local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained("NaturalAntibody/nanoBERT", local_files_only=True)
    unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=top_k)

    for pos, wt in enumerate(sequence):
        masked = list(sequence)
        masked[pos] = "<mask>"
        masked_seq = "".join(masked)

        try:
            predictions = unmasker(masked_seq)

            # look for wildtype prob in top_k
            wt_prob = None
            for pred in predictions:
                if pred["token_str"] == wt:
                    wt_prob = pred["score"]
                    break
            if wt_prob is None:
                wt_prob = 0.0

            for pred in predictions:
                mt = pred["token_str"]
                mut_prob = pred["score"]
                delta = mut_prob - wt_prob
                results.append({
                    "sample": sample_name,
                    "chain": chain_id,
                    "pos": pos + 1,
                    "wt": wt,
                    "mt": mt,
                    "mut_probability": mut_prob,
                    "wt_probability": wt_prob,
                    "delta_probability": delta
                })

        except Exception as e:
            print(f"Error at position {pos+1}: {e}")
            continue

    return pd.DataFrame(results)

def main():
    if len(sys.argv) != 5:
        print("Usage: python nanobert_worker.py <sample_name> <vh_sequence> <vl_sequence_or_NA> <format_type>")
        sys.exit(1)

    sample_name = sys.argv[1]
    vh_seq = sys.argv[2]
    vl_seq = sys.argv[3]  # not used, but kept for argument consistency
    format_type = sys.argv[4]

    if format_type != "Nanobody":
        print("ERROR: nanoBERT supports only Nanobody/VHH format.")
        sys.exit(1)

    print(f"DEBUG: running nanoBERT on sample '{sample_name}'")

    df = scan_nanobert(vh_seq, "H", sample_name, top_k=20)

    output_dir = "/home/eva/0_point_mutation/results/nanobert"
    os.makedirs(output_dir, exist_ok=True)
    # out_file = os.path.join(output_dir, f"{sample_name}_nanobert.csv")
    # df.to_csv(out_file, sep="\t", index=False)
    combined_csv = os.path.join(output_dir, f"{format_type}_nanobert.csv")
    df.to_csv(combined_csv, mode="a", index=False, header=False)
    print(f"Wrote nanoBERT tidy CSV to {combined_csv}")

if __name__ == "__main__":
    main()
