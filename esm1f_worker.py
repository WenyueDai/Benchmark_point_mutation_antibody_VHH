#!/usr/bin/env python3

import sys
import os
import pandas as pd
import torch
import esm

AAs = list("ACDEFGHIKLMNPQRSTVWY")

def main():
    if len(sys.argv) != 5:
        print("Usage: python esm1f_worker.py <sample_name> <vh_seq> <vl_seq> <format_type>")
        sys.exit(1)

    sample_name = sys.argv[1]
    vh_seq = sys.argv[2]
    vl_seq = sys.argv[3]
    format_type = sys.argv[4]

    print(f"DEBUG: running ESM1f on sample '{sample_name}', format={format_type}")

    # Load ESM-1b
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    records = []

    # process VH
    if vh_seq:
        data = [("VH", vh_seq)]
        _, _, tokens = batch_converter(data)
        with torch.no_grad():
            logits = model(tokens)["logits"][0]
        for pos, wt in enumerate(vh_seq, start=1):
            if wt not in AAs:
                continue
            wt_idx = alphabet.get_idx(wt)
            if wt_idx is None: continue
            wt_ll = logits[pos][wt_idx].item()
            for mt in AAs:
                mt_idx = alphabet.get_idx(mt)
                if mt_idx is None: continue
                mut_ll = logits[pos][mt_idx].item()
                delta = mut_ll - wt_ll
                records.append((
                    "VH", pos, wt, mt,
                    delta, mut_ll, wt_ll, sample_name
                ))

    # process VL if present and format_type is VHVL
    if vl_seq and format_type == "VHVL":
        data = [("VL", vl_seq)]
        _, _, tokens = batch_converter(data)
        with torch.no_grad():
            logits = model(tokens)["logits"][0]
        for pos, wt in enumerate(vl_seq, start=1):
            if wt not in AAs:
                continue
            wt_idx = alphabet.get_idx(wt)
            if wt_idx is None: continue
            wt_ll = logits[pos][wt_idx].item()
            for mt in AAs:
                mt_idx = alphabet.get_idx(mt)
                if mt_idx is None: continue
                mut_ll = logits[pos][mt_idx].item()
                delta = mut_ll - wt_ll
                records.append((
                    "VL", pos, wt, mt,
                    delta, mut_ll, wt_ll, sample_name
                ))

    tidy_df = pd.DataFrame(records, columns=[
        "chain", "pos", "wt", "mt",
        "delta_log_likelihood_esm1f",
        "mut_log_likelihood_esm1f",
        "wt_log_likelihood_esm1f",
        "sample"
    ])

    tidy_path = os.path.join(".", f"{sample_name}_esm1f_tidy.csv")
    tidy_df.to_csv(tidy_path, index=False)
    print(f"Wrote tidy ESM1f CSV to {tidy_path}")

    # append to combined tidy
    combined_csv = os.path.join(".", f"{format_type}_esm1f.csv")
    if not os.path.exists(combined_csv):
        tidy_df.to_csv(combined_csv, index=False)
        print(f"Created new combined tidy CSV: {combined_csv}")
    else:
        tidy_df.to_csv(combined_csv, mode="a", index=False, header=False)
        print(f"Appended to combined tidy CSV: {combined_csv}")

if __name__ == "__main__":
    main()
