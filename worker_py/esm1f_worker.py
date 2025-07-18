#!/usr/bin/env python3

import sys
import os
import pandas as pd
import torch
import esm
import time
import numpy as np
import argparse
import logging
from esm.inverse_folding.util import load_coords, score_sequence

sys.path.append("/home/eva/0_point_mutation/structural-evolution/bin")
from multichain_util import extract_coords_from_complex, score_sequence_in_complex
from util import load_structure

"""
Format: VHH — Input structure contains only the VHH (H chain). Mutate H, score H alone.
python esm1f_worker.py MyVHH EVQLVESGGGLVRVRTLPSEYTFWGQGTQVTVSS NA --mutate H --order H

Format: VHH + antigen — Input structure contains VHH (H) and antigen (e.g., A). Mutate H only, score the full complex (H + antigen).
python esm1f_worker.py VHH_complex EVQLVESGGGLVQPGGSLRLSCAASGRTFVRTLPSEYTFWGQGTQVTVSS NA --mutate H --order H,A

Format: VH/VL — Input structure contains VH (H) and VL (L). Mutate both H and L, score the VH–VL complex.
python esm1f_worker.py MyAb EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNLPSEYTFWGQGTQVTVSS EIVLTQSPATLSLSPGERAQAPRLLIYQPQQYNSYPWTFGQGTKLEIK --mutate H,L --order H,L

Format: VH/VL + antigen — Input structure contains VH (H), VL (L), and antigen (e.g., A). Mutate H and L, score the full complex (H + L + antigen).
python esm1f_worker.py MyAb EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNLPSEYTFWGQGTQVTVSS EIVLTQSPATLSLSPGERAQAPRLLIYQPQQYNSYPWTFGQGTKLEIK --mutate H,L --order H,L,A
"""

# ============================
# Logging setup
# ============================
logger = logging.getLogger("esm1f_logger")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
log_handler = None  # Assigned per sample in main()

AAs = list("ACDEFGHIKLMNPQRSTVWY")
ESM1F_OUTPUT_DIR = "results/esm1f"

def main():
    parser = argparse.ArgumentParser(
        description='ESM-IF1 mutation scan on protein complex.'
    )
    parser.add_argument('sample_name', type=str, help='Sample name (PDB file without extension)')
    parser.add_argument('vh_seq', type=str, help='VH sequence (optional)', nargs='?')
    parser.add_argument('vl_seq', type=str, help='VL sequence (optional)', nargs='?')
    parser.add_argument('--mutate', required=True, help='Comma-separated list of chains to mutate (e.g., H or H,L)')
    parser.add_argument('--order', default=None, help='Chain order override (e.g., H,M or H,L,M)')
    parser.add_argument('--nogpu', action='store_true', help='Disable GPU even if available')
    parser.add_argument('--format', type=str, help='Not used, for compatibility with other scripts')
    args = parser.parse_args()

    sample_name = args.sample_name
    chains_to_mutate = args.mutate.split(',')
    chain_order = args.order.split(',') if args.order else None

    # Setup log file
    global log_handler
    log_file = f"{sample_name}_esm1f.log"
    log_handler = logging.FileHandler(log_file, mode='w')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    logger.info(f"Starting ESM-IF1 scan on '{sample_name}', mutating chains {chains_to_mutate}")

    logger.info("Loading ESM-IF1 model...")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()

    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        logger.info("Model moved to GPU.")
    else:
        logger.info("Running on CPU.")

    pdb_path = os.path.join("pdbs", f"{sample_name}.pdb")
    if not os.path.exists(pdb_path):
        logger.error(f"Structure file not found at {pdb_path}")
        sys.exit(1)

    logger.info(f"Parsing PDB structure from {pdb_path}")
    try:
        structure = load_structure(pdb_path)
        coords_dict, native_seqs = extract_coords_from_complex(structure)
    except Exception as e:
        logger.error(f"Error loading structure: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(native_seqs)} chains: {', '.join(native_seqs.keys())}")
    logger.info("Scoring wild-type complex and each chain's sequence...")

    wt_ll_complex = None
    wt_ll_targets = {}

    for chain_id in chains_to_mutate:
        if chain_id not in native_seqs:
            logger.warning(f"Chain {chain_id} not found in structure. Skipping.")
            continue

        target_seq = native_seqs[chain_id]
        try:
            ll_complex, ll_target = score_sequence_in_complex(
                model, alphabet,
                coords_dict,
                native_seqs,
                target_chain_id=chain_id,
                target_seq=target_seq,
                order=chain_order
            )
        except Exception as e:
            logger.warning(f"Failed to score chain {chain_id}: {e}")
            continue

        if wt_ll_complex is None:
            wt_ll_complex = ll_complex
        wt_ll_targets[chain_id] = ll_target
        logger.info(f"WT LL - Complex: {ll_complex:.4f} | Chain {chain_id}: {ll_target:.4f}")

    if wt_ll_complex is None or not wt_ll_targets:
        logger.error("No valid wild-type log-likelihoods. Exiting.")
        sys.exit(1)

    logger.info("Beginning mutation scan...\n")
    records = []
    mut_counter = 0

    for chain_id in chains_to_mutate:
        if chain_id not in native_seqs:
            continue

        native_seq = native_seqs[chain_id]
        logger.info(f"Mutating chain {chain_id}, length {len(native_seq)}")

        for i, wt in enumerate(native_seq):
            if wt not in AAs:
                logger.info(f"Skipping non-canonical residue '{wt}' at position {i+1}")
                continue

            for mt in AAs:
                if mt == wt:
                    continue

                mut_seq = native_seq[:i] + mt + native_seq[i+1:]
                mutated_seqs = dict(native_seqs)
                mutated_seqs[chain_id] = mut_seq

                try:
                    mut_ll_complex, mut_ll_target = score_sequence_in_complex(
                        model, alphabet,
                        coords_dict,
                        mutated_seqs,
                        target_chain_id=chain_id,
                        target_seq=mut_seq,
                        order=chain_order
                    )

                    delta_complex = mut_ll_complex - wt_ll_complex
                    delta_target = mut_ll_target - wt_ll_targets[chain_id]

                    logger.info(
                        f"{wt}{i+1}{mt} | ΔLL (complex): {delta_complex:.4f} | ΔLL (chain {chain_id}): {delta_target:.4f}"
                    )

                    records.append((
                        sample_name, chain_id, i+1, wt, mt,
                        delta_complex, delta_target,
                        mut_ll_complex, mut_ll_target,
                        wt_ll_complex, wt_ll_targets[chain_id]
                    ))
                    mut_counter += 1

                    if mut_counter % 100 == 0:
                        logger.info(f"{mut_counter} mutations scored...")

                except Exception as e:
                    logger.warning(f"Skipped {wt}{i+1}{mt} (chain {chain_id}): {e}")
                    continue

    logger.info(f"Mutation scan complete: {mut_counter} mutations scored")

    # Save results
    df = pd.DataFrame(records, columns=[
        "sample", "chain", "pos", "wt", "mt",
        "delta_log_likelihood_complex_esm1f", "delta_log_likelihood_target_esm1f",
        "mt_log_likelihood_complex_esm1f", "mt_log_likelihood_target_esm1f",
        "wt_log_likelihood_complex_esm1f", "wt_log_likelihood_target_esm1f"
    ])
    out_path = os.path.join(ESM1F_OUTPUT_DIR, f"{sample_name}_esm1f_complex_tidy.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")

    # Cleanup logging
    logger.removeHandler(log_handler)
    log_handler.close()


if __name__ == "__main__":
    main()
