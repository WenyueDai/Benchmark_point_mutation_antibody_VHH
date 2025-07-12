#!/usr/bin/env python3

import sys
import os
import pandas as pd
import torch
import esm
import time
import numpy as np
from esm.inverse_folding.util import load_coords, score_sequence

AAs = list("ACDEFGHIKLMNPQRSTVWY")

def main():
    start_time = time.time()

    if len(sys.argv) != 5:
        print("Usage: python esm1f_worker.py <sample_name> <vh_seq> <vl_seq> <format_type>")
        sys.exit(1)

    sample_name = sys.argv[1]
    vh_seq = sys.argv[2]  # unused
    vl_seq = sys.argv[3]  # unused
    format_type = sys.argv[4].lower()

    print(f"DEBUG: Starting ESM1f run on sample '{sample_name}', format={format_type}")

    # Load ESM-IF1 model
    print("DEBUG: Loading ESM-IF1 model...")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()
    print("DEBUG: Model loaded.")

    if torch.cuda.is_available():
        model = model.cuda()
        print("INFO: Model moved to GPU.")
    else:
        print("INFO: Running on CPU.")

    # Determine which chain(s) to score
    if format_type == "nanobody":
        chain_id = "H"
    elif format_type == "vhvl":
        chain_id = "H,L"
    else:
        print(f"ERROR: Unsupported format_type '{format_type}'. Use 'Nanobody' or 'VHVL'.")
        sys.exit(1)

    # Load structure
    pdb_path = os.path.join("./pdbs", f"{sample_name}.pdb")
    if not os.path.exists(pdb_path):
        print(f"ERROR: Structure file not found at {pdb_path}")
        sys.exit(1)

    print(f"DEBUG: Loading structure from {pdb_path}")
    try:
        coords_list, wt_seq = load_coords(pdb_path, chain=chain_id)
        print(f"DEBUG: Loaded coords for chain(s) {chain_id}, sequence length = {len(wt_seq)}")
    except Exception as e:
        print(f"ERROR loading structure: {e}")
        sys.exit(1)

    print("DEBUG: Scoring wild-type sequence...")
    try:
        wt_ll, _ = score_sequence(model, alphabet, coords_list, wt_seq)
        print(f"DEBUG: Wild-type log-likelihood = {wt_ll:.4f}")
    except Exception as e:
        print(f"ERROR during wild-type scoring: {e}")
        sys.exit(1)

    print("DEBUG: Starting mutation scan...")
    records = []
    mut_counter = 0

    for i, wt in enumerate(wt_seq):
        if wt not in AAs:
            print(f"WARNING: Skipping non-standard residue '{wt}' at position {i + 1}")
            continue

        if (i + 1) % 10 == 0:
            print(f"DEBUG: Processing position {i + 1}/{len(wt_seq)}")

        for mt in AAs:
            if mt == wt:
                continue

            mut_seq = wt_seq[:i] + mt + wt_seq[i + 1:]
            try:
                mut_ll, _ = score_sequence(model, alphabet, coords_list, mut_seq)
                delta = mut_ll - wt_ll
                records.append((
                    chain_id, i + 1, wt, mt,
                    delta, mut_ll, wt_ll, sample_name
                ))
                mut_counter += 1

                if mut_counter % 100 == 0:
                    print(f"DEBUG: Completed {mut_counter} mutations")

            except Exception as e:
                print(f"WARNING: Skipped mutation {wt}{i+1}{mt} due to error: {e}")
                continue

    print(f"INFO: Finished mutation scan with {mut_counter} total mutations")

    # Save tidy result
    tidy_df = pd.DataFrame(records, columns=[
        "chain", "pos", "wt", "mt",
        "delta_log_likelihood_esm1f",
        "mut_log_likelihood_esm1f",
        "wt_log_likelihood_esm1f",
        "sample"
    ])

    tidy_path = os.path.join(".", f"{sample_name}_esm1f_tidy.csv")
    tidy_df.to_csv(tidy_path, index=False)
    print(f"INFO: Wrote tidy CSV to {tidy_path}")

    combined_csv = os.path.join(".", f"{format_type}_esm1f.csv")
    if not os.path.exists(combined_csv):
        tidy_df.to_csv(combined_csv, index=False)
        print(f"INFO: Created new combined CSV: {combined_csv}")
    else:
        tidy_df.to_csv(combined_csv, mode="a", index=False, header=False)
        print(f"INFO: Appended to existing combined CSV: {combined_csv}")

    print(f"INFO: Completed in {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()
