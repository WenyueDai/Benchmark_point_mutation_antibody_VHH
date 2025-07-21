#!/usr/bin/env python3

import sys
import os
import pandas as pd
import torch
import torch.nn.functional as F
import esm
import logging
import argparse

logger = logging.getLogger("esm1v_logger")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
log_handler = None

AAs = list("ACDEFGHIKLMNPQRSTVWY")

def main():
    parser = argparse.ArgumentParser(description='ESM-1v mutation scan on protein complex.')
    parser.add_argument('sample_name', type=str, help='Sample name (PDB file without extension)')
    parser.add_argument('vh_seq', type=str, help='VH sequence (optional)', nargs='?')
    parser.add_argument('vl_seq', type=str, help='VL sequence (optional)', nargs='?')
    parser.add_argument('--format', type=str, help='Format type (e.g., VHVL, Nanobody)')
    parser.add_argument('--log_likelihood_only', action='store_true',
                        help='Only compute WT log-likelihood, skip mutation scan')
    args = parser.parse_args()

    sample_name = args.sample_name
    vh_seq = args.vh_seq
    vl_seq = args.vl_seq
    format_type = args.format
    log_likelihood_only = args.log_likelihood_only

    global log_handler
    log_file = f"{sample_name}_esm1v.log"
    log_handler = logging.FileHandler(log_file, mode='w')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    logger.info(f"Running ESM-1v on sample '{sample_name}', format={format_type}, log_likelihood_only={log_likelihood_only}")

    model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to GPU.")
    else:
        logger.info("Running on CPU.")

    records = []
    total_ll = 0.0

    # Process VH
    if vh_seq:
        if log_likelihood_only:
            total_ll += compute_total_log_likelihood(vh_seq, model, alphabet, batch_converter)
        else:
            records.extend(compute_mutation_log_likelihoods(vh_seq, "VH", model, alphabet, batch_converter, sample_name))
        logger.info(f"Processed VH sequence.")

    # Process VL
    if vl_seq and format_type == "VHVL":
        if log_likelihood_only:
            total_ll += compute_total_log_likelihood(vl_seq, model, alphabet, batch_converter)
        else:
            records.extend(compute_mutation_log_likelihoods(vl_seq, "VL", model, alphabet, batch_converter, sample_name))
        logger.info(f"Processed VL sequence.")

    output_dir = "results/esm1v"
    os.makedirs(output_dir, exist_ok=True)

    if log_likelihood_only:
        df = pd.DataFrame([{
            "sample": sample_name,
            "log_likelihood_esm1v": total_ll
        }])
        output_file = os.path.join(output_dir, f"{format_type}_esm1v_likelihood_only.csv")
        write_header = not os.path.exists(output_file)
        df.to_csv(output_file, index=False, mode="a", header=write_header)
        print(f"Written WT-only log-likelihood to: {output_file}")
        return

    df = pd.DataFrame(records, columns=[
        "chain", "pos", "wt", "mt",
        "delta_seq_log_likelihood_esm1v",
        "mut_log_likelihood_esm1v",
        "wt_log_likelihood_esm1v",
        "sample"
    ])
    combined_csv = os.path.join(output_dir, f"{format_type}_esm1v.csv")
    if not os.path.exists(combined_csv):
        df.to_csv(combined_csv, index=False)
        print(f"Created new CSV: {combined_csv}")
    else:
        df.to_csv(combined_csv, mode="a", header=False, index=False)
        print(f"Appended to CSV: {combined_csv}")


def compute_mutation_log_likelihoods(seq, chain_name, model, alphabet, batch_converter, sample_name):
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


def compute_total_log_likelihood(seq, model, alphabet, batch_converter):
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    if torch.cuda.is_available():
        tokens = tokens.cuda()

    with torch.no_grad():
        logits = model(tokens)["logits"][0]
        log_probs = F.log_softmax(logits, dim=-1)

    total_ll = 0.0
    for pos, wt in enumerate(seq, start=1):
        if wt in AAs:
            idx = alphabet.get_idx(wt)
            if idx is not None:
                total_ll += log_probs[pos][idx].item()
    return total_ll


if __name__ == "__main__":
    main()
