#!/usr/bin/env python3
"""
Comprehensive debugging script for protein design pipeline tensor size mismatch
Usage: python debug_protein_pipeline.py <sample_name>
"""

import sys
import os
import torch
import pandas as pd
from byprot.utils.config import compose_config as Cfg
from byprot.tasks.fixedbb.designer import Designer
import traceback
from collections import defaultdict

# Configuration
PDB_INPUT_DIR = "/home/eva/0_point_mutation/pdbs"
LM_MODEL_PATH = "/home/eva/0_point_mutation/ByProt-main/checkpoint/lm_design_esm2_650m"
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

class ProteinDebugger:
    def __init__(self, sample_name):
        self.sample_name = sample_name
        self.pdb_path = os.path.join(PDB_INPUT_DIR, f"{sample_name}.pdb")
        self.debug_info = defaultdict(dict)
        self.designer = None
        
    def print_section(self, title):
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
    
    def print_step(self, step, info=""):
        print(f"📋 {step}")
        if info:
            print(f"   {info}")
    
    def check_file_exists(self):
        self.print_section("FILE EXISTENCE CHECK")
        
        self.print_step("Checking PDB file", self.pdb_path)
        if not os.path.exists(self.pdb_path):
            print(f"❌ ERROR: PDB file does not exist!")
            return False
        
        print(f"✅ PDB file exists")
        
        # Check file size and basic content
        file_size = os.path.getsize(self.pdb_path)
        self.debug_info['file']['size'] = file_size
        print(f"📊 File size: {file_size} bytes")
        
        # Read first few lines
        with open(self.pdb_path, 'r') as f:
            lines = f.readlines()[:10]
            print(f"📊 First 10 lines:")
            for i, line in enumerate(lines):
                print(f"   {i+1}: {line.strip()}")
        
        return True
    
    def analyze_pdb_structure(self):
        self.print_section("PDB STRUCTURE ANALYSIS")
        
        atom_count = 0
        residue_count = 0
        chain_info = defaultdict(list)
        
        with open(self.pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_count += 1
                    chain_id = line[21]
                    res_num = int(line[22:26].strip())
                    res_name = line[17:20].strip()
                    
                    if res_num not in chain_info[chain_id]:
                        chain_info[chain_id].append((res_num, res_name))
        
        self.debug_info['pdb']['atom_count'] = atom_count
        self.debug_info['pdb']['chains'] = dict(chain_info)
        
        print(f"📊 Total ATOM records: {atom_count}")
        print(f"📊 Chains found: {list(chain_info.keys())}")
        
        for chain_id, residues in chain_info.items():
            residues.sort()  # Sort by residue number
            print(f"📊 Chain {chain_id}: {len(residues)} residues")
            print(f"   First 5: {residues[:5]}")
            print(f"   Last 5: {residues[-5:]}")
            
            # Check for gaps
            res_numbers = [r[0] for r in residues]
            gaps = []
            for i in range(len(res_numbers)-1):
                if res_numbers[i+1] - res_numbers[i] > 1:
                    gaps.append((res_numbers[i], res_numbers[i+1]))
            
            if gaps:
                print(f"⚠️  Gaps found in chain {chain_id}: {gaps}")
            
            self.debug_info['pdb'][f'chain_{chain_id}_residues'] = len(residues)
            self.debug_info['pdb'][f'chain_{chain_id}_gaps'] = gaps
    
    def initialize_designer(self):
        self.print_section("DESIGNER INITIALIZATION")
        
        try:
            cfg = Cfg(
                cuda=True,
                generator=Cfg(
                    max_iter=3,
                    strategy="denoise",
                    temperature=0,
                    eval_sc=False,
                )
            )
            
            self.print_step("Creating Designer instance")
            self.designer = Designer(experiment_path=LM_MODEL_PATH, cfg=cfg)
            print("✅ Designer created successfully")
            
            self.print_step("Loading PDB structure")
            self.designer.set_structure(self.pdb_path)
            print("✅ Structure loaded")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR during designer initialization: {e}")
            traceback.print_exc()
            return False
    
    def analyze_loaded_structure(self):
        self.print_section("LOADED STRUCTURE ANALYSIS")
        
        if not self.designer or not self.designer._structure:
            print("❌ No structure loaded")
            return False
        
        structure = self.designer._structure
        
        print(f"📊 Structure keys: {list(structure.keys())}")
        
        # Analyze sequence
        if "seq" in structure:
            seq = structure["seq"]
            if isinstance(seq, tuple):
                print(f"📊 Sequence is tuple with {len(seq)} elements")
                for i, s in enumerate(seq):
                    print(f"   Element {i}: length {len(s)}, first 20: {s[:20]}")
                    self.debug_info['structure'][f'seq_{i}_length'] = len(s)
                seq = seq[0]  # Use first element
            else:
                print(f"📊 Sequence type: {type(seq)}")
            
            print(f"📊 Sequence length: {len(seq)}")
            print(f"📊 First 50 chars: {seq[:50]}")
            print(f"📊 Last 50 chars: {seq[-50:]}")
            
            # Check for non-standard amino acids
            non_standard = [aa for aa in seq if aa not in AMINO_ACIDS]
            if non_standard:
                print(f"⚠️  Non-standard amino acids found: {set(non_standard)}")
                print(f"⚠️  Count: {len(non_standard)}")
            
            self.debug_info['structure']['seq_length'] = len(seq)
            self.debug_info['structure']['non_standard_aa'] = list(set(non_standard))
        
        # Analyze other structure components
        for key, value in structure.items():
            if key != "seq":
                if isinstance(value, torch.Tensor):
                    print(f"📊 {key}: tensor shape {value.shape}")
                    self.debug_info['structure'][f'{key}_shape'] = list(value.shape)
                elif isinstance(value, (list, tuple)):
                    print(f"📊 {key}: {type(value)} length {len(value)}")
                    self.debug_info['structure'][f'{key}_length'] = len(value)
                else:
                    print(f"📊 {key}: {type(value)}")
        
        return True
    
    def test_featurization(self):
        self.print_section("FEATURIZATION TESTING")
        
        if not self.designer:
            print("❌ No designer available")
            return False
        
        try:
            self.print_step("Running featurization")
            batch = self.designer._featurize()
            
            print(f"📊 Batch type: {type(batch)}")
            print(f"📊 Batch keys: {list(batch.keys())}")
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"📊 {key}: tensor shape {value.shape}, dtype {value.dtype}")
                    self.debug_info['featurization'][f'{key}_shape'] = list(value.shape)
                    self.debug_info['featurization'][f'{key}_dtype'] = str(value.dtype)
                else:
                    print(f"📊 {key}: {type(value)}")
            
            return batch
            
        except Exception as e:
            print(f"❌ ERROR during featurization: {e}")
            traceback.print_exc()
            return None
    
    def test_model_forward(self, batch):
        self.print_section("MODEL FORWARD PASS TESTING")
        
        if not batch:
            print("❌ No batch available")
            return False
        
        try:
            self.print_step("Testing model forward pass")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.no_grad():
                output = self.designer.model(batch)
            
            print(f"📊 Model output type: {type(output)}")
            
            if isinstance(output, dict):
                print(f"📊 Output keys: {list(output.keys())}")
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        print(f"📊 {key}: tensor shape {value.shape}")
                        self.debug_info['model_output'][f'{key}_shape'] = list(value.shape)
            elif isinstance(output, torch.Tensor):
                print(f"📊 Output tensor shape: {output.shape}")
                self.debug_info['model_output']['tensor_shape'] = list(output.shape)
            
            return output
            
        except Exception as e:
            print(f"❌ ERROR during model forward pass: {e}")
            traceback.print_exc()
            
            # Try to get more specific error info
            if "size" in str(e).lower():
                print("\n🔍 TENSOR SIZE MISMATCH DETECTED!")
                print("🔍 This suggests input tensors don't match model expectations")
                
                # Print tensor shapes for debugging
                print("\n📊 Input tensor shapes:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"   {key}: {value.shape}")
            
            return None
    
    def compare_with_working_sample(self, working_sample=""):
        self.print_section("COMPARISON WITH WORKING SAMPLE")
        
        if not working_sample:
            print("⚠️  No working sample provided for comparison")
            return
        
        working_path = os.path.join(PDB_INPUT_DIR, f"{working_sample}.pdb")
        if not os.path.exists(working_path):
            print(f"⚠️  Working sample PDB not found: {working_path}")
            return
        
        print(f"📊 Comparing {self.sample_name} with {working_sample}")
        
        # Quick comparison of file sizes
        current_size = os.path.getsize(self.pdb_path)
        working_size = os.path.getsize(working_path)
        
        print(f"📊 File sizes - Current: {current_size}, Working: {working_size}")
        
        # You could add more detailed comparison here
    
    def generate_debug_report(self):
        self.print_section("DEBUG REPORT SUMMARY")
        
        print("📋 Key Findings:")
        
        # Structure vs PDB comparison
        if 'structure' in self.debug_info and 'pdb' in self.debug_info:
            struct_len = self.debug_info['structure'].get('seq_length', 'Unknown')
            pdb_residues = sum(v for k, v in self.debug_info['pdb'].items() if k.endswith('_residues'))
            
            print(f"📊 Structure sequence length: {struct_len}")
            print(f"📊 PDB residue count: {pdb_residues}")
            
            if struct_len != pdb_residues and struct_len != 'Unknown':
                print(f"⚠️  LENGTH MISMATCH: Structure ({struct_len}) vs PDB ({pdb_residues})")
        
        # Featurization output
        if 'featurization' in self.debug_info:
            print(f"📊 Featurization successful: {len(self.debug_info['featurization'])} tensors created")
        
        # Model compatibility
        if 'model_output' in self.debug_info:
            print(f"📊 Model forward pass: SUCCESS")
        else:
            print(f"❌ Model forward pass: FAILED")
        
        # Save debug info to file
        debug_file = f"debug_report_{self.sample_name}.txt"
        with open(debug_file, 'w') as f:
            f.write(f"Debug Report for {self.sample_name}\n")
            f.write("="*50 + "\n\n")
            
            for category, data in self.debug_info.items():
                f.write(f"{category.upper()}:\n")
                for key, value in data.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"📁 Debug report saved to: {debug_file}")
    
    def run_full_debug(self):
        """Run complete debugging sequence"""
        print(f"🐛 Starting full debug for sample: {self.sample_name}")
        
        # Step 1: Check file existence
        if not self.check_file_exists():
            return False
        
        # Step 2: Analyze PDB structure
        self.analyze_pdb_structure()
        
        # Step 3: Initialize designer
        if not self.initialize_designer():
            return False
        
        # Step 4: Analyze loaded structure
        if not self.analyze_loaded_structure():
            return False
        
        # Step 5: Test featurization
        batch = self.test_featurization()
        
        # Step 6: Test model forward pass
        if batch:
            self.test_model_forward(batch)
        
        # Step 7: Generate report
        self.generate_debug_report()
        
        return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_protein_pipeline.py <sample_name> [working_sample]")
        sys.exit(1)
    
    sample_name = sys.argv[1]
    working_sample = sys.argv[2] if len(sys.argv) > 2 else ""
    
    debugger = ProteinDebugger(sample_name)
    
    try:
        debugger.run_full_debug()
        
        if working_sample:
            debugger.compare_with_working_sample(working_sample)
            
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()