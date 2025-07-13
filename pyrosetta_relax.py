import os
import sys
from pyrosetta import init, pose_from_pdb, create_score_function
from pyrosetta.rosetta.core.scoring import ScoreType
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.relax import FastRelax

def relax_pdb(pdb_path, n_repeats=5):
    init("-relax:cartesian -relax:constrain_relax_to_start_coords "
         "-relax:coord_constrain_sidechains -ex1 -ex2 -use_input_sc -mute all")

    # Backup original
    backup_path = pdb_path.replace(".pdb", "_before_relax.pdb")
    os.rename(pdb_path, backup_path)
    print(f"Original PDB renamed to: {backup_path}")

    # Load pose
    pose = pose_from_pdb(backup_path)

    # Score function with coordinate constraints
    scorefxn = create_score_function("ref2015_cart.wts")
    scorefxn.set_weight(ScoreType.coordinate_constraint, 1.0)

    print(f"Starting score: {scorefxn(pose):.3f}")

    # MoveMap: relax side chains only
    movemap = MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)

    # Relax with correct constructor
    relax = FastRelax(scorefxn, n_repeats)
    relax.set_movemap(movemap)
    relax.cartesian(True)  # Enable Cartesian relax, to allow high-resolution refinement
    relax.minimize_bond_angles(True)
    relax.minimize_bond_lengths(True)

    relax.apply(pose)

    print(f"Final score: {scorefxn(pose):.3f}")

    # Save result
    pose.dump_pdb(pdb_path)
    print(f"Relaxed structure saved to: {pdb_path}")

if __name__ == "__main__":
    pdb_path = sys.argv[1]
    n_repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    relax_pdb(pdb_path, n_repeats)
