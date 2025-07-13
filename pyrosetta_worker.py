#!/usr/bin/env python3

import sys
import os
import pandas as pd
from pyrosetta import init, pose_from_pdb, Pose, get_fa_scorefxn
from pyrosetta.rosetta.core.select.residue_selector import (
    ResidueIndexSelector, NeighborhoodResidueSelector, NotResidueSelector
)
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import (
    InitializeFromCommandline, IncludeCurrent, NoRepackDisulfides,
    PreventRepackingRLT, RestrictToRepackingRLT,
    OperateOnResidueSubset, RestrictAbsentCanonicalAASRLT
)
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.protocols.relax import FastRelax

# Initialize PyRosetta quietly
init(extra_options='-mute all')

# Canonical amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def relax_pose(pose):
    """Apply FastRelax to minimize structure before mutations."""
    scorefxn = get_fa_scorefxn()
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.constrain_relax_to_start_coords(True)
    relax.apply(pose)
    return pose

def pack_mutate(pose, posi, mutant_aa):
    """Mutate a single residue using TaskFactory and PackRotamersMover."""
    scorefxn = get_fa_scorefxn()
    test_pose = Pose()
    test_pose.assign(pose)

    mut_selector = ResidueIndexSelector()
    mut_selector.set_index(posi)

    nbr_selector = NeighborhoodResidueSelector()
    nbr_selector.set_focus_selector(mut_selector)
    nbr_selector.set_include_focus_in_subset(True)

    not_design = NotResidueSelector(mut_selector)

    tf = TaskFactory()
    tf.push_back(InitializeFromCommandline())
    tf.push_back(IncludeCurrent())
    tf.push_back(NoRepackDisulfides())
    tf.push_back(OperateOnResidueSubset(PreventRepackingRLT(), nbr_selector, True))
    tf.push_back(OperateOnResidueSubset(RestrictToRepackingRLT(), not_design))

    restrict_aa = RestrictAbsentCanonicalAASRLT()
    restrict_aa.aas_to_keep(mutant_aa)
    tf.push_back(OperateOnResidueSubset(restrict_aa, mut_selector))

    packer = PackRotamersMover(scorefxn)
    packer.task_factory(tf)
    packer.apply(test_pose)

    return test_pose

def compute_ddg(pdb_path, chain_id):
    """Compute ΔΔG for each mutation in a specific chain."""
    pose = pose_from_pdb(pdb_path)
    pose = relax_pose(pose)
    scorefxn = get_fa_scorefxn()

    results = []

    # Detect valid chain IDs
    valid_chains = {chr(pose.residue(i).chain()) for i in range(1, pose.total_residue() + 1) if pose.residue(i).is_protein()}
    if chain_id not in valid_chains:
        print(f"[WARNING] Chain '{chain_id}' not found. Available chains: {valid_chains}")
        if valid_chains:
            chain_id = sorted(valid_chains)[0]
            print(f"[INFO] Falling back to chain: '{chain_id}'")

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not res.is_protein():
            continue
        if chr(res.chain()) != chain_id:
            continue

        wt = res.name1()
        wt_score = scorefxn(pose)

        for aa in AMINO_ACIDS:
            if aa == wt:
                continue

            try:
                mut_pose = pack_mutate(pose, i, aa)
                mut_score = scorefxn(mut_pose)
                ddg = mut_score - wt_score

                results.append({
                    "pos": i,
                    "chain": chain_id,
                    "wt": wt,
                    "mt": aa,
                    "wt_score": wt_score,
                    "mut_score": mut_score,
                    "ddG": ddg,
                    "delta_neg_ddG_pyrosetta": -ddg  # for log-likelihood comparisons
                })

            except Exception as e:
                print(f"Skipping mutation {wt}{i}{aa} on chain {chain_id}: {e}")

    return pd.DataFrame(results)

def assign_chain_from_region(df):
    """Fix chain ID based on region if it is non-printable (e.g., \x01)."""
    def region_to_chain(region):
        return "H" if region == "VH" else "L" if region == "VL" else "?"

    df["chain"] = df.apply(
        lambda row: region_to_chain(row["region"]) if ord(str(row["chain"])[0]) < 32 else row["chain"],
        axis=1
    )
    return df

def main():
    if len(sys.argv) != 5:
        print("Usage: python pyrosetta_worker.py <sample_name> <vh_seq> <vl_seq> <format_type>")
        sys.exit(1)

    sample_name = sys.argv[1]
    vh_seq = sys.argv[2]
    vl_seq = sys.argv[3]
    format_type = sys.argv[4]

    pdb_path = f"/home/eva/0_point_mutation/pdbs/{sample_name}.pdb"
    output_dir = f"/home/eva/0_point_mutation/results/pyrosetta"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(pdb_path):
        print(f"ERROR: PDB file not found: {pdb_path}")
        sys.exit(1)

    all_results = []

    if format_type == "Nanobody":
        print(f"Running PyRosetta ddG for Nanobody (VH only)")
        df_vh = compute_ddg(pdb_path, chain_id="H")
        df_vh["region"] = "VH"
        df_vh["name"] = sample_name
        df_vh = assign_chain_from_region(df_vh)
        all_results.append(df_vh)

    elif format_type == "VHVL":
        print(f"Running PyRosetta ddG for VHVL (VH + VL)")
        df_vh = compute_ddg(pdb_path, chain_id="H")
        df_vh["region"] = "VH"
        df_vh["name"] = sample_name  # Add sample name
        df_vh = assign_chain_from_region(df_vh)
        all_results.append(df_vh)

        df_vl = compute_ddg(pdb_path, chain_id="L")
        df_vl["region"] = "VL"
        df_vl["name"] = sample_name
        df_vl = assign_chain_from_region(df_vl)
        all_results.append(df_vl)

    else:
        print(f"ERROR: format_type must be 'Nanobody' or 'VHVL', got '{format_type}'")
        sys.exit(1)

    combined = pd.concat(all_results, ignore_index=True)
    output_path = os.path.join(output_dir, f"{sample_name}_pyrosetta.csv")
    combined.to_csv(output_path, index=False)
    print(f"Saved PyRosetta ddG results to: {output_path}")

if __name__ == "__main__":
    main()
