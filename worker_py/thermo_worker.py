#!/usr/bin/env python3

import sys
import os
import pandas as pd
import torch
import esm
import time
import numpy as np
import keras

AAs = list("ACDEFGHIKLMNPQRSTVWY")
print(f"INFO: TensorFlow version: {tf.__version__}")
print(f"INFO: TensorFlow running on GPU: {tf.config.list_physical_devices('GPU')}")

def generate_embedding(sequence, model, alphabet, batch_converter):
    model.eval()
    batch_labels, batch_strs, batch_tokens = batch_converter([("sequence", sequence)])
    batch_tokens = batch_tokens.to(next(model.parameters()).device)
    print(f"DEBUG: Input tokens moved to device: {batch_tokens.device}")
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    seq_len = (batch_tokens != alphabet.padding_idx).sum(1).item()
    embedding = token_representations[0, 1:seq_len - 1].mean(0)  # exclude BOS/EOS
    return embedding.cpu().numpy()

def main():
    start_time = time.time()

    if len(sys.argv) != 5:
        print("Usage: python tempro_worker.py <sample_name> <vh_seq> <vl_seq> <format_type>")
        sys.exit(1)

    sample_name = sys.argv[1]
    vh_seq = sys.argv[2]  # treated as full input sequence
    vl_seq = sys.argv[3]  # unused
    format_type = sys.argv[4].lower()

    print(f"DEBUG: Starting TEMPRO run on sample '{sample_name}', format={format_type}")

    # Load TEMPRO model
    print("DEBUG: Loading TEMPRO model...")
    ann_model = keras.models.load_model("/home/eva/0_point_mutation/TEMPRO-main/user/saved_ANNmodels_1500epoch/ESM_650M.h5", compile=False)
    print("DEBUG: Model loaded.")

    # Load ESM-2 model
    print("DEBUG: Loading ESM2-650M model...")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    device = next(esm_model.parameters()).device
    print(f"INFO: ESM model running on device: {device}")
    print("DEBUG: ESM model loaded.")

    if torch.cuda.is_available():
        esm_model = esm_model.cuda()
        print("INFO: ESM model moved to GPU.")
        
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        reserved_mem = torch.cuda.memory_reserved(0) / 1e9
        allocated_mem = torch.cuda.memory_allocated(0) / 1e9
        print(f"DEBUG: CUDA total memory: {total_mem:.2f} GB")
        print(f"DEBUG: CUDA reserved memory: {reserved_mem:.2f} GB")
        print(f"DEBUG: CUDA allocated memory: {allocated_mem:.2f} GB")
    else:
        print("INFO: Running on CPU.")

    wt_seq = vh_seq  # Use vh_seq as the input sequence
    print(f"DEBUG: Wild-type sequence length = {len(wt_seq)}")

    print("DEBUG: Predicting wild-type melting temperature...")
    wt_embedding = generate_embedding(wt_seq, esm_model, alphabet, batch_converter)
    wt_tm = ann_model.predict(np.expand_dims(wt_embedding, axis=0))[0][0]
    print(f"DEBUG: Wild-type predicted Tm = {wt_tm:.2f}")

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
            mut_seq = wt_seq[:i] + mt + wt_seq[i + 1:]
            try:
                mut_embedding = generate_embedding(mut_seq, esm_model, alphabet, batch_converter)
                mut_tm = ann_model.predict(np.expand_dims(mut_embedding, axis=0))[0][0]
                delta_tm = mut_tm - wt_tm

                records.append((
                    "H", i + 1, wt, mt,  # assuming nanobody format
                    delta_tm, mut_tm, wt_tm, sample_name
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
        "delta_tm",
        "mut_tm",
        "wt_tm",
        "sample"
    ])

    tidy_path = os.path.join(".", f"{sample_name}_tempro_tidy.csv")
    tidy_df.to_csv(tidy_path, index=False)
    print(f"INFO: Wrote tidy CSV to {tidy_path}")

    combined_csv = os.path.join(".", f"{format_type}_tempro.csv")
    if not os.path.exists(combined_csv):
        tidy_df.to_csv(combined_csv, index=False)
        print(f"INFO: Created new combined CSV: {combined_csv}")
    else:
        tidy_df.to_csv(combined_csv, mode="a", index=False, header=False)
        print(f"INFO: Appended to existing combined CSV: {combined_csv}")

    print(f"INFO: Completed in {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()
