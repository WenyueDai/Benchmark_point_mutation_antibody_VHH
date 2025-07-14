#!/usr/bin/env python3

import sys
import os
import gemmi
import subprocess
import pandas as pd

PDB_INPUT_DIR = "/home/eva/0_point_mutation/pdbs"  # Adjust if needed
ANTIFOLD_OUTPUT_DIR = "/home/eva/0_point_mutation/results/antifold"

def main():
    if len(sys.argv) != 5:
        print("Usage: python antifold_worker.py <sample_name> <vh_seq> <vl_seq_or_NA> <format_type>")
        sys.exit(1)

    sample_name = sys.argv[1]
    vh_seq = sys.argv[2]   # Ignored here
    vl_seq = sys.argv[3]   # Ignored here
    format_type = sys.argv[4]

    print(f"DEBUG: running antifold on sample '{sample_name}', format={format_type}")

    pdb_file = os.path.join(PDB_INPUT_DIR, f"{sample_name}.pdb")
    cif_file = os.path.join(ANTIFOLD_OUTPUT_DIR, f"{sample_name}.cif")

    if not os.path.exists(pdb_file):
        print(f"ERROR: missing PDB file: {pdb_file}")
        sys.exit(1)

    if not os.path.exists(cif_file):
        print(f"Converting {pdb_file} to mmCIF with gemmi...")
        try:
            st = gemmi.read_structure(pdb_file)
            st.setup_entities()
            st.make_mmcif_document().write_file(cif_file)
            print(f"Converted to {cif_file}")
        except Exception as e:
            print(f"ERROR during CIF conversion: {e}")
            print("Falling back to using the PDB directly.")
            cif_file = pdb_file
    else:
        print(f"Using existing CIF: {cif_file}")

    try:
        structure = gemmi.read_structure(cif_file)
        chains = [chain.name for chain in structure[0]]
    except Exception as e:
        print(f"ERROR reading structure from {cif_file}: {e}")
        sys.exit(1)

    if format_type == "Nanobody":
        chain = 'H'
        print(f"Detected nanobody chain: {chain}")
        cmd = [
            "python", "-m", "antifold.main",
            "--pdb_file", cif_file,
            "--nanobody_chain", chain,
            "--out_dir", ANTIFOLD_OUTPUT_DIR
        ]
    elif format_type == "VHVL":
        if len(chains) < 2:
            print(f"ERROR: expected at least 2 chains for VHVL, found: {chains}")
            sys.exit(1)
        heavy_chain = chains[0]
        light_chain = chains[1]
        print(f"Detected heavy chain: {heavy_chain}, light chain: {light_chain}")
        cmd = [
            "python", "-m", "antifold.main",
            "--pdb_file", cif_file,
            "--heavy_chain", heavy_chain,
            "--light_chain", light_chain,
            "--out_dir", ANTIFOLD_OUTPUT_DIR
        ]
    else:
        print("ERROR: Only 'Nanobody' and 'VHVL' format types are supported.")
        sys.exit(1)

    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Antifold run complete.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR running antifold: {e}")
        sys.exit(1)

    # tidy csv transform
    try:
        available_csvs = [
            os.path.join(ANTIFOLD_OUTPUT_DIR, f)
            for f in os.listdir(ANTIFOLD_OUTPUT_DIR)
            if f.startswith(sample_name) and f.endswith(".csv")
        ]

        if not available_csvs:
            print(f"ERROR: Could not find any AntiFold CSV output files for {sample_name}")
            sys.exit(1)

        output_columns = [
            "chain", "pos", "wt", "mt",
            "mut_log_likelihood_antifold",
            "wt_log_likelihood_antifold",
            "delta_log_likelihood_antifold",
            "sample"
        ]

        for antifold_csv in available_csvs:
            print(f"Parsing AntiFold CSV to tidy format: {antifold_csv}")
            af = pd.read_csv(antifold_csv)
            aas = list("ACDEFGHIKLMNPQRSTVWY")

            records = []
            for _, row in af.iterrows():
                pos = row["pdb_pos"]
                chain_label = row["pdb_chain"]
                wt = row["pdb_res"]
                wt_ll = row[wt] if wt in aas else None
                if wt_ll is None:
                    continue
                for mt in aas:
                    mut_ll = row[mt]
                    delta = mut_ll - wt_ll
                    records.append((
                        chain_label, pos, wt, mt, mut_ll, wt_ll, delta, sample_name
                    ))

            tidy_df = pd.DataFrame(records, columns=output_columns)
            tidy_path = os.path.join(ANTIFOLD_OUTPUT_DIR, f"{sample_name}_tidy.csv")
            tidy_df.to_csv(tidy_path, index=False, sep="\t")
            print(f"Wrote tidy AntiFold-style CSV to {tidy_path}")

            combined_tidy = os.path.join(ANTIFOLD_OUTPUT_DIR, f"{format_type}_antifold.csv")
            if not os.path.exists(combined_tidy):
                tidy_df.to_csv(combined_tidy, index=False, sep="\t", columns=output_columns)
                print(f"Created new combined tidy CSV: {combined_tidy}")
            else:
                tidy_df.to_csv(combined_tidy, mode="a", header=False, index=False, sep="\t", columns=output_columns)
                print(f"Appended to combined tidy CSV: {combined_tidy}")

    except Exception as e:
        print(f"ERROR while transforming tidy CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
