import os
import numpy as np
import glob

# Path to your raw data
DATA_DIR = r"C:\Oak Ridge\Model\data\raw"


def scan_labels():
    label_files = glob.glob(os.path.join(DATA_DIR, "*_labels.txt"))

    if not label_files:
        print("ERROR: No label files found!")
        return

    global_min = float('inf')
    global_max = float('-inf')
    has_nans = False

    for f_path in label_files:
        filename = os.path.basename(f_path)
        try:
            # Read file
            arr = np.loadtxt(f_path)

            # Handle 1D vs 2D arrays (CloudCompare format)
            if arr.ndim == 1:
                labels = arr
            else:
                labels = arr[:, -1]  # Last column

            # Check for NaNs
            if np.isnan(labels).any():
                print(f"[FAIL] {filename} contains NaNs (Not a Number)!")
                has_nans = True

            # Check for Inf
            if np.isinf(labels).any():
                print(f"[FAIL] {filename} contains Infinite values!")

            # Check range and Ignore NaNs for range check
            unique_vals = np.unique(labels[~np.isnan(labels)])
            min_val = unique_vals.min()
            max_val = unique_vals.max()

            global_min = min(global_min, min_val)
            global_max = max(global_max, max_val)

            if min_val < 0 or max_val >= 4:
                print(
                    f"[FAIL] {filename}: Found rogue labels {unique_vals} (Expected 0-3)")
            else:
                print(f"[OK]   {filename}: {unique_vals}")

        except Exception as e:
            print(f"[ERR]  {filename}: Read error - {e}")

    print("\n" + "="*30)
    print(f"Global Range Found: [{global_min}, {global_max}]")
    if has_nans:
        print("CRITICAL: NaNs found.")
    if global_max >= 4 or global_min < 0:
        print("CRITICAL: Labels outside 0-3 found.")


if __name__ == "__main__":
    scan_labels()
