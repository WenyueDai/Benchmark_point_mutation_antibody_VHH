#!/usr/bin/env python3

import sys
import os
import pandas as pd
import torch
import torch.nn.functional as F
import esm
import logging
import argparse

# ============================
# Logging setup
# ============================
logger = logging.getLogger("esm1v_logger")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
log_handler = None  # Assigned per sample in main()

AAs = list("ACDEFGHIKLMNPQRSTVWY")

def main():
    parser = argparse.ArgumentParser(
        description='ESM-1v mutation scan on protein complex.'
    )
    parser.add_argument('sample_name', type=str, help='Sample name (PDB file without extension)')
    parser.add_argument('vh_seq', type=str, help='VH sequence (optional)', nargs='?')
    parser.add_argument('vl_seq', type=str, help='VL sequence (optional)', nargs='?')
    parser.add_argument('--format', type=str, help='Format type (e.g., VHVL, Nanobody)')
    args = parser.parse_args()
    sample_name = args.sample_name
    vh_seq = args.vh_seq
    vl_seq = args.vl_seq
    format_type = args.format
    
    # Setup log file
    global log_handler
    log_file = f"{sample_name}_esm1v.log"
    log_handler = logging.FileHandler(log_file, mode='w')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    
    logger.info(f"Running ESM-1v on sample '{sample_name}', format={format_type}")
    # Load ESM-1v model
    model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to GPU.")
    else:
        logger.info("Running on CPU.")
    
    records = []
    
    # Process VH sequence
    if vh_seq:
        records.extend(
            compute_mutation_log_likelihoods(vh_seq, "VH", model, alphabet, batch_converter, sample_name)
        )
    logger.info(f"Processed VH sequence: {vh_seq}")
        
    # Process VL sequence if present and format_type is VHVL
    if vl_seq and format_type == "VHVL":
        records.extend(
            compute_mutation_log_likelihoods(vl_seq, "VL", model, alphabet, batch_converter, sample_name)
        )
        
    logger.info(f"Processed VL sequence: {vl_seq}")
        
    # Final output (no tidy, only <format_type>_esm1v.csv as requested)
    combined_csv = os.path.join("results/esm1v", f"{format_type}_esm1v.csv")
    df = pd.DataFrame(records, columns=[
        "chain", "pos", "wt", "mt",
        "delta_seq_log_likelihood_esm1v",
        "mut_log_likelihood_esm1v",
        "wt_log_likelihood_esm1v",
        "sample"
    ])
    
    if not os.path.exists(combined_csv):
        df.to_csv(combined_csv, index=False)
        print(f"Created new combined tidy CSV: {combined_csv}")
    else:
        df.to_csv(combined_csv, mode="a", index=False, header=False)
        print(f"Appended to combined tidy CSV: {combined_csv}")
        

def compute_mutation_log_likelihoods(seq, chain_name, model, alphabet, batch_converter, sample_name):
    """
    Compute mutation log-likelihoods and sequence delta for all single mutations in `seq`.
    Returns a list of records (rows) for the final CSV.
    """
    data = [(chain_name, seq)]
    _, _, tokens = batch_converter(data)
    if torch.cuda.is_available():
        tokens = tokens.cuda()

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

if __name__ == "__main__":
    main()