#!/usr/bin/env python3

import sys
import os
import pandas as pd
import argparse
from transformers import pipeline, RobertaTokenizer, AutoModelForMaskedLM
import torch
import math

def scan_nanobert(sequence, chain_id, sample_name, top_k=20, log_likelihood_only=False):
    """
    For a given sequence, systematically mask each position and get top_k predictions
    from nanoBERT. Also computes wt_probability, mut_probability, and delta_probability.
    If log_likelihood_only=True, computes a pseudo log-likelihood score.
    """
    results = []

    # Load nanoBERT locally
    tokenizer = RobertaTokenizer.from_pretrained("NaturalAntibody/nanoBERT", local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained("NaturalAntibody/nanoBERT", local_files_only=True)
    unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=top_k)

    log_likelihood_sum = 0.0
    valid_pos_count = 0

    for pos, wt in enumerate(sequence):
        masked = list(sequence)
        masked[pos] = "<mask>"
        masked_seq = "".join(masked)

        try:
            predictions = unmasker(masked_seq)

            # Find wildtype probability
            wt_prob = 0.0
            for pred in predictions:
                if pred["token_str"] == wt:
                    wt_prob = pred["score"]
                    break

            # log-likelihood approximation
            if log_likelihood_only:
                if wt_prob > 0:
                    log_likelihood_sum += math.log(wt_prob)
                    valid_pos_count += 1
                continue  # no need to collect mutation table

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

    if log_likelihood_only:
        return log_likelihood_sum, valid_pos_count
    else:
        return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_name")
    parser.add_argument("vh_sequence")
    parser.add_argument("vl_sequence")  # not used, but retained for compatibility
    parser.add_argument("--format", required=True)
    parser.add_argument("--log_likelihood_only", action="store_true")
    args = parser.parse_args()

    if args.format != "Nanobody":
        print("ERROR: nanoBERT supports only Nanobody/VHH format.")
        sys.exit(1)

    print(f"DEBUG: Running nanoBERT on '{args.sample_name}' (log_likelihood_only={args.log_likelihood_only})")

    output_dir = "/home/eva/0_point_mutation/results/nanobert"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.format}_nanobert.csv")

    if args.log_likelihood_only:
        log_likelihood, count = scan_nanobert(args.vh_sequence, "H", args.sample_name, log_likelihood_only=True)
        avg_ll = log_likelihood / count if count > 0 else float("-inf")
        df = pd.DataFrame([{
            "sample": args.sample_name,
            "log_likelihood_nanobert": avg_ll,
            "positions_scanned": count
        }])
        df.to_csv(output_file, mode="a", index=False, header=not os.path.exists(output_file))
    else:
        df = scan_nanobert(args.vh_sequence, "H", args.sample_name, top_k=20, log_likelihood_only=False)
        df.to_csv(output_file, mode="a", index=False, header=not os.path.exists(output_file))

    print(f"Results written to: {output_file}")

if __name__ == "__main__":
    main()
