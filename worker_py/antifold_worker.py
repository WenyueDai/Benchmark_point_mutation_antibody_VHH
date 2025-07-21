#!/usr/bin/env python3

import sys
import os
import argparse
import logging
import subprocess
import gemmi
import pandas as pd

PDB_INPUT_DIR = "/home/eva/0_point_mutation/pdbs"
ANTIFOLD_OUTPUT_DIR = "/home/eva/0_point_mutation/results/antifold"
AAs = list("ACDEFGHIKLMNPQRSTVWY")

# Logging setup
logger = logging.getLogger("antifold_logger")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
log_handler = None

def convert_to_cif(pdb_file, cif_file):
    try:
        st = gemmi.read_structure(pdb_file)
        st.setup_entities()
        st.make_mmcif_document().write_file(cif_file)
        logger.info(f"Converted {pdb_file} to {cif_file}")
        return cif_file
    except Exception as e:
        logger.error(f"Failed to convert to mmCIF: {e}")
        return pdb_file

def run_antifold(cif_file, format_type):
    try:
        structure = gemmi.read_structure(cif_file)
        chains = [chain.name for chain in structure[0]]
        logger.info(f"Detected chains in structure: {chains}")
    except Exception as e:
        logger.error(f"Could not read CIF structure: {e}")
        sys.exit(1)

    if format_type.lower() == "nanobody":
        if len(chains) < 1:
            logger.error("Nanobody format requires at least one chain.")
            sys.exit(1)
        chain = chains[0]
        logger.info(f"Running AntiFold on nanobody chain: {chain}")
        cmd = [
            "python", "-m", "antifold.main",
            "--pdb_file", cif_file,
            "--nanobody_chain", chain,
            "--out_dir", ANTIFOLD_OUTPUT_DIR
        ]
    elif format_type.lower() == "vhvl":
        if len(chains) < 2:
            logger.error("VHVL format requires at least two chains.")
            sys.exit(1)
        heavy, light = chains[0], chains[1]
        logger.info(f"Running AntiFold on VH/VL chains: H={heavy}, L={light}")
        cmd = [
            "python", "-m", "antifold.main",
            "--pdb_file", cif_file,
            "--heavy_chain", heavy,
            "--light_chain", light,
            "--out_dir", ANTIFOLD_OUTPUT_DIR
        ]
    else:
        logger.error(f"Unsupported format type: {format_type}")
        sys.exit(1)

    logger.info(f"Executing AntiFold: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info("AntiFold run complete.")

def parse_antifold_csv(sample_name, format_type, log_likelihood_only=False):
    try:
        available_csvs = [
            os.path.join(ANTIFOLD_OUTPUT_DIR, f)
            for f in os.listdir(ANTIFOLD_OUTPUT_DIR)
            if f.startswith(sample_name) and f.endswith(".csv")
        ]

        if not available_csvs:
            print(f"ERROR: Could not find any AntiFold CSV output files for {sample_name}")
            sys.exit(1)

        for antifold_csv in available_csvs:
            print(f"Parsing AntiFold CSV to tidy format: {antifold_csv}")
            af = pd.read_csv(antifold_csv)

            if log_likelihood_only:
                total_ll = 0.0
                count = 0

                for _, row in af.iterrows():
                    wt = row["pdb_res"]
                    if wt in AAs and wt in af.columns:
                        total_ll += row[wt]
                        count += 1

                df = pd.DataFrame([{
                    "sample": sample_name,
                    "log_likelihood_antifold": total_ll
                }])

                output_file = os.path.join(ANTIFOLD_OUTPUT_DIR, f"{format_type}_antifold_likelihood_only.csv")
                write_header = not os.path.exists(output_file)
                df.to_csv(output_file, mode="a", header=write_header, index=False)
                print(f"WT-only log-likelihood written to: {output_file}")
                return

            else:
                records = []
                for _, row in af.iterrows():
                    pos = row["pdb_pos"]
                    chain_label = row["pdb_chain"]
                    wt = row["pdb_res"]
                    if wt not in AAs or wt not in af.columns:
                        logger.warning(f"Skipping row at pos {pos} with unknown wt residue: {wt}")
                        continue
                    wt_ll = row[wt]
                    for mt in AAs:
                        mut_ll = row[mt]
                        delta = mut_ll - wt_ll
                        records.append((
                            chain_label, pos, wt, mt, mut_ll, wt_ll, delta, sample_name
                        ))

                tidy_df = pd.DataFrame(records, columns=[
                    "chain", "pos", "wt", "mt",
                    "mut_log_likelihood_antifold",
                    "wt_log_likelihood_antifold",
                    "delta_log_likelihood_antifold",
                    "sample"
                ])

                combined_tidy = os.path.join(ANTIFOLD_OUTPUT_DIR, f"{format_type}_antifold.csv")
                write_header = not os.path.exists(combined_tidy)
                tidy_df.to_csv(combined_tidy, mode="a", header=write_header, index=False, sep="\t")
                print(f"{'Created' if write_header else 'Appended to'} combined tidy CSV: {combined_tidy}")

    except Exception as e:
        print(f"ERROR while transforming tidy CSV: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AntiFold mutation scan using PDB input")
    parser.add_argument("sample_name", type=str, help="Sample name (PDB filename without extension)")
    parser.add_argument("vh_seq", type=str, help="VH sequence (ignored)")
    parser.add_argument("vl_seq", type=str, help="VL sequence or 'NA' (ignored)")
    parser.add_argument("--mutate", required=True, help="Comma-separated list of chains to mutate (e.g., H,L)")
    parser.add_argument("--format", required=True, choices=["Nanobody", "VHVL"], help="Input format type")
    parser.add_argument('--log_likelihood_only', action='store_true', help='Only compute WT log-likelihood, skip mutation scan')

    args = parser.parse_args()
    sample_name = args.sample_name
    chains_to_mutate = args.mutate.split(",")
    format_type = args.format

    # Setup log file
    global log_handler
    log_file = os.path.join(ANTIFOLD_OUTPUT_DIR, f"{sample_name}_antifold.log")
    log_handler = logging.FileHandler(log_file, mode="w")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    logger.info(f"Starting AntiFold scan on {sample_name} with format: {format_type}")
    logger.info(f"Chains to mutate: {chains_to_mutate}")

    pdb_file = os.path.join(PDB_INPUT_DIR, f"{sample_name}.pdb")
    cif_file = os.path.join(ANTIFOLD_OUTPUT_DIR, f"{sample_name}.cif")

    if not os.path.exists(pdb_file):
        logger.error(f"PDB file not found: {pdb_file}")
        sys.exit(1)

    if not os.path.exists(cif_file):
        cif_file = convert_to_cif(pdb_file, cif_file)
    else:
        logger.info(f"Using existing CIF: {cif_file}")

    try:
        run_antifold(cif_file, format_type)
        parse_antifold_csv(sample_name, format_type, log_likelihood_only=args.log_likelihood_only)
    except subprocess.CalledProcessError as e:
        logger.error(f"AntiFold subprocess failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

    logger.info("AntiFold mutation scan complete.")
    logger.removeHandler(log_handler)
    log_handler.close()

if __name__ == "__main__":
    main()
