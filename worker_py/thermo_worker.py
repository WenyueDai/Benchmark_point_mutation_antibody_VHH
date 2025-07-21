#!/usr/bin/env python3

import sys
import os
import time
import argparse
import pandas as pd
import torch
import numpy as np
import esm
import keras

AAs = list("ACDEFGHIKLMNPQRSTVWY")

def generate_embedding(sequence, model, alphabet, batch_converter):
    model.eval()
    batch_labels, batch_strs, batch_tokens = batch_converter([("sequence", sequence)])
    batch_tokens = batch_tokens.to(next(model.parameters()).device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    seq_len = (batch_tokens != alphabet.padding_idx).sum(1).item()
    embedding = token_representations[0, 1:seq_len - 1].mean(0)  # exclude BOS/EOS
    return embedding.cpu().numpy()

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("sample_name", type=str)
    parser.add_argument("vh_seq", type=str)
    parser.add_argument("vl_seq", type=str)  # unused
    parser.add_argument("--format", type=str, required=True)
    parser.add_argument("--log_likelihood_only", action="store_true", help="Only compute wild-type Tm")
    args = parser.parse_args()

    sample_name = args.sample_name
    wt_seq = args.vh_seq  # Nanobody assumed
    format_type = args.format.lower()

    print(f"DEBUG: Starting TEMPRO run on '{sample_name}', format={format_type}, log_likelihood_only={args.log_likelihood_only}")

    # Load TEMPRO and ESM models
    ann_model = keras.models.load_model(
        "/home/eva/0_point_mutation/TEMPRO-main/user/saved_ANNmodels_1500epoch/ESM_650M.h5",
        compile=False
    )
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()

    if torch.cuda.is_available():
        esm_model = esm_model.cuda()
        print("INFO: ESM model moved to GPU.")
    else:
        print("INFO: Running ESM model on CPU.")

    # Compute WT Tm
    wt_embedding = generate_embedding(wt_seq, esm_model, alphabet, batch_converter)
    wt_tm = ann_model.predict(np.expand_dims(wt_embedding, axis=0))[0][0]
    print(f"DEBUG: Wild-type predicted Tm = {wt_tm:.2f}")

    # If only computing WT Tm, skip mutation scan
    if args.log_likelihood_only:
        df = pd.DataFrame([{"sample": sample_name, "wt_tm": wt_tm}])
        out_path = os.path.join(".", f"{format_type}_tempro.csv")
        df.to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path))
        print(f"INFO: Wrote WT Tm to {out_path}")
        print(f"INFO: Completed in {time.time() - start_time:.1f} seconds")
        return

    # Mutation scan
    records = []
    mut_counter = 0

    for i, wt in enumerate(wt_seq):
        if wt not in AAs:
            print(f"WARNING: Skipping non-standard residue '{wt}' at position {i + 1}")
            continue

        for mt in AAs:
            mut_seq = wt_seq[:i] + mt + wt_seq[i + 1:]
            try:
                mut_embedding = generate_embedding(mut_seq, esm_model, alphabet, batch_converter)
                mut_tm = ann_model.predict(np.expand_dims(mut_embedding, axis=0))[0][0]
                delta_tm = mut_tm - wt_tm

                records.append((
                    "H", i + 1, wt, mt,
                    delta_tm, mut_tm, wt_tm, sample_name
                ))
                mut_counter += 1

                if mut_counter % 100 == 0:
                    print(f"DEBUG: Completed {mut_counter} mutations")

            except Exception as e:
                print(f"WARNING: Skipped mutation {wt}{i+1}{mt} due to error: {e}")
                continue

    tidy_df = pd.DataFrame(records, columns=[
        "chain", "pos", "wt", "mt",
        "delta_tm", "mut_tm", "wt_tm", "sample"
    ])

    tidy_path = os.path.join(".", f"{sample_name}_tempro_tidy.csv")
    tidy_df.to_csv(tidy_path, index=False)
    print(f"INFO: Wrote tidy mutation scan CSV to {tidy_path}")

    combined_csv = os.path.join(".", f"{format_type}_tempro.csv")
    tidy_df.to_csv(combined_csv, mode="a", index=False, header=not os.path.exists(combined_csv))
    print(f"INFO: Appended results to: {combined_csv}")
    print(f"INFO: Total runtime: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()
