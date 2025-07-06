#!/usr/bin/env python3

import sys
import os
import gemmi
import subprocess

def main():
    if len(sys.argv) != 5:
        print("Usage: python antifold_worker.py <sample_name> <pdb_dir> <output_dir> <format_type>")
        sys.exit(1)

    sample_name = sys.argv[1]
    pdb_dir = sys.argv[2]
    output_dir = sys.argv[3]
    format_type = sys.argv[4]

    print(f"DEBUG: running antifold on sample '{sample_name}', format={format_type}")

    pdb_file = os.path.join(pdb_dir, f"{sample_name}.pdb")
    cif_file = os.path.join(output_dir, f"{sample_name}.cif")

    # check PDB existence
    if not os.path.exists(pdb_file):
        print(f"ERROR: missing PDB file: {pdb_file}")
        sys.exit(1)

    # try conversion to CIF
    if not os.path.exists(cif_file):
        print(f"Converting {pdb_file} to mmCIF with gemmi...")
        try:
            st = gemmi.read_structure(pdb_file)
            st.setup_entities()
            st.write_minimal_cif(cif_file)
            print(f"✅ Converted to {cif_file}")
        except Exception as e:
            print(f"ERROR during CIF conversion: {e}")
            print("⚠️  Falling back to using the PDB directly.")
            cif_file = pdb_file  # fallback
    else:
        print(f"✅ Using existing CIF: {cif_file}")

    # check structure to get chain information
    try:
        structure = gemmi.read_structure(cif_file)
        chains = [chain.name for chain in structure[0]]
    except Exception as e:
        print(f"ERROR reading structure from {cif_file}: {e}")
        sys.exit(1)

    if format_type == "Nanobody":
        if len(chains) != 1:
            print(f"ERROR: expected exactly 1 chain for Nanobody, found: {chains}")
            sys.exit(1)
        chain = chains[0]
        print(f"✅ Detected nanobody chain: {chain}")
        cmd = [
            "python", "-m", "antifold.main",
            "--pdb_file", cif_file,
            "--nanobody_chain", chain,
            "--out_dir", output_dir
        ]
    elif format_type == "VHVL":
        if len(chains) < 2:
            print(f"ERROR: expected at least 2 chains for VHVL, found: {chains}")
            sys.exit(1)
        heavy_chain = chains[0]
        light_chain = chains[1]
        print(f"✅ Detected heavy chain: {heavy_chain}, light chain: {light_chain}")
        cmd = [
            "python", "-m", "antifold.main",
            "--pdb_file", cif_file,
            "--heavy_chain", heavy_chain,
            "--light_chain", light_chain,
            "--out_dir", output_dir
        ]
    else:
        print("ERROR: Only 'Nanobody' and 'VHVL' format types are supported.")
        sys.exit(1)

    # run antifold
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("✅ Antifold run complete.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR running antifold: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
