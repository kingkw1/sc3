"""
Quick Test Script
=================

This script performs a quick test to ensure:
1. Required libraries are installed
2. Data files are accessible
3. Basic parsing works

Run this before running the full analyses to catch any setup issues.
"""

import os
import sys

def test_imports():
    """Test if required libraries are available."""
    print("Testing imports...")
    try:
        import numpy as np
        print("  ✓ numpy imported successfully")
    except ImportError:
        print("  ✗ numpy not found. Install with: pip install numpy")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("  ✓ matplotlib imported successfully")
    except ImportError:
        print("  ✗ matplotlib not found. Install with: pip install matplotlib")
        return False
    
    try:
        from scipy.ndimage import gaussian_filter
        print("  ✓ scipy imported successfully")
    except ImportError:
        print("  ✗ scipy not found. Install with: pip install scipy")
        return False
    
    return True

def test_data_access():
    """Test if data directories are accessible."""
    print("\nTesting data access...")
    
    sim_dir = '../data/sim'
    web_dir = '../data/web'
    
    if not os.path.exists(sim_dir):
        print(f"  ✗ Simulation data directory not found: {sim_dir}")
        print(f"    Current directory: {os.getcwd()}")
        print(f"    Make sure you're running from the analysis/ directory")
        return False
    else:
        sim_dirs = [d for d in os.listdir(sim_dir) if os.path.isdir(os.path.join(sim_dir, d))]
        print(f"  ✓ Simulation directory found with {len(sim_dirs)} simulations")
    
    if not os.path.exists(web_dir):
        print(f"  ✗ Web data directory not found: {web_dir}")
        return False
    else:
        web_sessions = [d for d in os.listdir(web_dir) if os.path.isdir(os.path.join(web_dir, d))]
        print(f"  ✓ Web directory found with {len(web_sessions)} sessions")
    
    return True

def test_json_parsing():
    """Test if we can parse a sample JSON file."""
    print("\nTesting JSON parsing...")
    
    import json
    
    sim_dir = '../data/sim'
    sim_dirs = [d for d in os.listdir(sim_dir) if os.path.isdir(os.path.join(sim_dir, d))]
    
    if not sim_dirs:
        print("  ✗ No simulation directories found")
        return False
    
    test_sim = sim_dirs[0]
    config_path = os.path.join(sim_dir, test_sim, 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"  ✓ Successfully parsed config.json")
        print(f"    Scenario: {config.get('name', 'Unknown')}")
        print(f"    Entities: {len(config.get('entities', []))}")
        print(f"    Max time: {config.get('max_scenario_time', 0)/3600:.1f} hours")
        return True
    except Exception as e:
        print(f"  ✗ Failed to parse config.json: {e}")
        return False

def test_output_directory():
    """Test if output directory can be created."""
    print("\nTesting output directory...")
    
    output_dir = './outputs'
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to write a test file
        test_file = os.path.join(output_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        
        os.remove(test_file)
        
        print(f"  ✓ Output directory ready: {os.path.abspath(output_dir)}")
        return True
    except Exception as e:
        print(f"  ✗ Cannot create output directory: {e}")
        return False

def main():
    print("=" * 60)
    print("SC3 ANALYSIS - SETUP TEST")
    print("=" * 60)
    
    tests = [
        ("Import Libraries", test_imports),
        ("Data Access", test_data_access),
        ("JSON Parsing", test_json_parsing),
        ("Output Directory", test_output_directory),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 60)
        result = test_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to run the analyses.")
        print("\nNext steps:")
        print("  1. Run individual scripts: python 1_casualties_over_time.py")
        print("  2. Or run all at once: python run_all_analyses.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above before proceeding.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
