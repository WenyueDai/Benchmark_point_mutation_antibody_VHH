#!/usr/bin/env python3

import sys
import os
import pandas as pd
import torch
import torch.nn.functional as F
import esm

AAs = list("ACDEFGHIKLMNPQRSTVWY")

def compute_mutation_log_likelihoods(seq, chain_name, model, alphabet, batch_converter, sample_name):
    """
    Compute mutation log-likelihoods and sequence delta for all single mutations in `seq`.
    Returns a list of records (rows) for the final CSV.
    """
    data = [(chain_name, seq)]
    _, _, tokens = batch_converter(data)

    with torch.no_grad():
        logits = model(tokens)["logits"][0]
        log_probs = F.log_softmax(logits, dim=-1)

    records = []
    for pos, wt in enumerate(seq, start=1):
        if wt not in AAs:
            continue
        wt_idx = alphabet.get_idx(wt)
        if wt_idx is None:
            continue
        wt_ll = log_probs[pos][wt_idx].item()

        for mt in AAs:
            mt_idx = alphabet.get_idx(mt)
            if mt_idx is None:
                continue
            mut_ll = log_probs[pos][mt_idx].item()
            delta = mut_ll - wt_ll
            records.append((
                chain_name, pos, wt, mt,
                delta, mut_ll, wt_ll, sample_name
            ))
    return records

def main():
    if len(sys.argv) != 5:
        print("Usage: python esm2_worker.py <sample_name> <vh_seq> <vl_seq> <format_type>")
        sys.exit(1)

    sample_name = sys.argv[1]
    vh_seq = sys.argv[2]
    vl_seq = sys.argv[3]
    format_type = sys.argv[4]

    print(f"DEBUG: running ESM2 on sample '{sample_name}', format={format_type}")

    # Load ESM1v
    model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    records = []

    # process VH
    if vh_seq:
        records.extend(
            compute_mutation_log_likelihoods(vh_seq, "VH", model, alphabet, batch_converter, sample_name)
        )

    # process VL if present and format_type is VHVL
    if vl_seq and format_type == "VHVL":
        records.extend(
            compute_mutation_log_likelihoods(vl_seq, "VL", model, alphabet, batch_converter, sample_name)
        )

    # Final output (no tidy, only <format_type>_esm2.csv as requested)
    combined_csv = os.path.join(".", f"{format_type}_esm2.csv")
    df = pd.DataFrame(records, columns=[
        "chain", "pos", "wt", "mt",
        "delta_seq_log_likelihood",
        "mut_log_likelihood",
        "wt_log_likelihood",
        "sample"
    ])

    if not os.path.exists(combined_csv):
        df.to_csv(combined_csv, index=False)
        print(f"Created new combined tidy CSV: {combined_csv}")
    else:
        df.to_csv(combined_csv, mode="a", index=False, header=False)
        print(f"Appended to combined tidy CSV: {combined_csv}")

if __name__ == "__main__":
    main()
