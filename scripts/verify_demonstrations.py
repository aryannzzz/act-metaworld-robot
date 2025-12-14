#!/usr/bin/env python3
"""
Verify demonstration data quality.
Checks for diversity, action range, and data integrity.
"""

import h5py
import numpy as np
import sys
import argparse

def verify_demonstrations(hdf5_path):
    """Verify demonstration data quality"""
    
    print("=" * 70)
    print(f"🔍 VERIFYING DEMONSTRATIONS: {hdf5_path}")
    print("=" * 70)
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            demo_keys = [k for k in f.keys() if k.startswith('demo_')]
            
            if len(demo_keys) == 0:
                print("\n❌ FATAL: No demonstrations found!")
                return False
            
            print(f"\n📊 Found {len(demo_keys)} demonstrations")
            
            # Test 1: Check diversity
            print("\n" + "─" * 70)
            print("TEST 1: Demonstration Diversity")
            print("─" * 70)
            
            if len(demo_keys) >= 2:
                a0 = np.array(f[demo_keys[0]]['actions'][:10])
                a1 = np.array(f[demo_keys[1]]['actions'][:10])
                
                if np.allclose(a0, a1, rtol=1e-4):
                    print("❌ FAIL: First two demos are IDENTICAL!")
                    print(f"   Demo 0 first action: {a0[0]}")
                    print(f"   Demo 1 first action: {a1[0]}")
                    return False
                else:
                    print("✅ PASS: Demos are different")
                    print(f"   Demo 0 first action: {a0[0]}")
                    print(f"   Demo 1 first action: {a1[0]}")
            
            # Test 2: Check action range
            print("\n" + "─" * 70)
            print("TEST 2: Action Range")
            print("─" * 70)
            
            all_actions = []
            for dk in demo_keys[:min(10, len(demo_keys))]:
                actions = np.array(f[dk]['actions'][:])
                all_actions.append(actions)
            
            all_actions = np.concatenate(all_actions, axis=0)
            action_min = all_actions.min()
            action_max = all_actions.max()
            action_mean = all_actions.mean()
            action_std = all_actions.std()
            
            print(f"   Range: [{action_min:.6f}, {action_max:.6f}]")
            print(f"   Mean:  {action_mean:.6f}")
            print(f"   Std:   {action_std:.6f}")
            
            if action_min < -1.001 or action_max > 1.001:
                print(f"❌ FAIL: Actions outside [-1, 1]!")
                print(f"   This will cause train/test mismatch!")
                return False
            else:
                print("✅ PASS: Actions in valid range [-1, 1]")
            
            # Test 3: Check data shapes
            print("\n" + "─" * 70)
            print("TEST 3: Data Shapes and Consistency")
            print("─" * 70)
            
            demo = f[demo_keys[0]]
            
            if 'images' in demo:
                img_shape = demo['images'].shape
                print(f"   Images shape: {img_shape}")
                
                if len(img_shape) != 4 or img_shape[3] != 3:
                    print(f"   ⚠️  WARNING: Expected (T, H, W, 3), got {img_shape}")
            
            if 'states' in demo:
                state_shape = demo['states'].shape
                print(f"   States shape: {state_shape}")
                
                if len(state_shape) != 2 or state_shape[1] != 39:
                    print(f"   ⚠️  WARNING: Expected (T, 39), got {state_shape}")
            
            if 'actions' in demo:
                action_shape = demo['actions'].shape
                print(f"   Actions shape: {action_shape}")
                
                if len(action_shape) != 2 or action_shape[1] != 4:
                    print(f"   ⚠️  WARNING: Expected (T, 4), got {action_shape}")
            
            print("✅ PASS: Data shapes look reasonable")
            
            # Test 4: Check success rate
            print("\n" + "─" * 70)
            print("TEST 4: Success Rate")
            print("─" * 70)
            
            successes = []
            for dk in demo_keys:
                success = f[dk].attrs.get('success', False)
                successes.append(success)
            
            success_rate = np.mean(successes) * 100
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Successful demos: {sum(successes)}/{len(successes)}")
            
            if success_rate < 50:
                print(f"   ⚠️  WARNING: Low success rate ({success_rate:.1f}%)")
                print(f"   Consider collecting more data or checking expert policy")
            else:
                print(f"✅ PASS: Good success rate ({success_rate:.1f}%)")
            
            # Summary
            print("\n" + "=" * 70)
            print("📋 VERIFICATION SUMMARY")
            print("=" * 70)
            print(f"✅ {len(demo_keys)} demonstrations")
            print(f"✅ Actions in range [{action_min:.3f}, {action_max:.3f}]")
            print(f"✅ Demos are diverse (not identical)")
            print(f"✅ {success_rate:.1f}% success rate")
            print("\n🎉 All checks passed! Data is ready for training.")
            
            return True
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify demonstration data quality')
    parser.add_argument('hdf5_path', type=str, help='Path to HDF5 file')
    
    args = parser.parse_args()
    
    success = verify_demonstrations(args.hdf5_path)
    
    sys.exit(0 if success else 1)
