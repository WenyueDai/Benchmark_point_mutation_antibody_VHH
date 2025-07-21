#!/usr/bin/env python3

import os
import sys
import argparse
import logging
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

# Initialize PyRosetta
init(extra_options='-mute all')

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PYROSETTA_OUTPUT_DIR = "results/pyrosetta"

logger = logging.getLogger("pyrosetta_ddg")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

def setup_logger(sample_name):
    os.makedirs(PYROSETTA_OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(PYROSETTA_OUTPUT_DIR, f"{sample_name}_pyrosetta.log")
    log_handler = logging.FileHandler(log_file, mode='w')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    return log_handler

def pack_mutate(pose, posi, mutant_aa):
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

def get_chain_id(pose, residue_index):
    return pose.pdb_info().chain(residue_index)

def compute_ddg_all(pdb_path, chains_to_mutate, sample_name):
    pose = pose_from_pdb(pdb_path)
    scorefxn = get_fa_scorefxn()
    wt_score = scorefxn(pose)
    logger.info(f"WT total score: {wt_score:.3f}")

    results = []

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        chain = get_chain_id(pose, i)
        if not res.is_protein() or chain not in chains_to_mutate:
            continue
        wt = res.name1()
        for mt in AMINO_ACIDS:
            if mt == wt:
                continue
            try:
                mut_pose = pack_mutate(pose, i, mt)
                mut_score = scorefxn(mut_pose)
                ddg = mut_score - wt_score
                logger.info(f"{wt}{i}{mt} | ΔG: {ddg:.3f}")
                results.append({
                    "sample": sample_name,
                    "chain": chain,
                    "pos": i,
                    "wt": wt,
                    "mt": mt,
                    "delta_dG_complex_pyrosetta": -ddg,
                    "mut_dG_complex_pyrosetta": mut_score,
                    "wt_dG_complex_pyrosetta": wt_score,
                })
            except Exception as e:
                logger.warning(f"Skipped {wt}{i}{mt} on chain {chain}: {e}")
                continue

    return pd.DataFrame(results)

def compute_wt_score_only(pdb_path, sample_name):
    pose = pose_from_pdb(pdb_path)
    scorefxn = get_fa_scorefxn()
    wt_score = scorefxn(pose)
    logger.info(f"[log_likelihood_only] WT score for {sample_name}: {wt_score:.3f}")
    return pd.DataFrame([{
        "sample": sample_name,
        "wt_dG_complex_pyrosetta": wt_score
    }])

def main():
    parser = argparse.ArgumentParser(description="PyRosetta mutation scan")
    parser.add_argument("sample_name", type=str)
    parser.add_argument("vh_seq", type=str)
    parser.add_argument("vl_seq", type=str)
    parser.add_argument("--mutate", required=True)
    parser.add_argument("--order", default=None)
    parser.add_argument("--nogpu", action="store_true")
    parser.add_argument("--format", type=str)
    parser.add_argument("--log_likelihood_only", action="store_true", help="Skip mutation scan and compute WT score only")
    args = parser.parse_args()

    sample_name = args.sample_name
    chains_to_mutate = args.mutate.split(",")

    log_handler = setup_logger(sample_name)
    pdb_path = os.path.join("pdbs", f"{sample_name}.pdb")
    if not os.path.exists(pdb_path):
        logger.error(f"PDB file not found: {pdb_path}")
        sys.exit(1)

    logger.info(f"Running PyRosetta on '{sample_name}' with log_likelihood_only={args.log_likelihood_only}")

    if args.log_likelihood_only:
        df = compute_wt_score_only(pdb_path, sample_name)
    else:
        df = compute_ddg_all(pdb_path, chains_to_mutate, sample_name)

    os.makedirs(PYROSETTA_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(PYROSETTA_OUTPUT_DIR, f"{sample_name}_pyrosetta_tidy.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved results to: {out_path}")

    logger.removeHandler(log_handler)
    log_handler.close()

if __name__ == "__main__":
    main()
