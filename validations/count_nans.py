import os
import numpy as np
import glob

DATA_DIR = r"C:\Oak Ridge\Model\data\raw"


def count_nans():
    files = glob.glob(os.path.join(DATA_DIR, "*_labels.txt"))

    total_nans_all_files = 0

    for f_path in files:
        filename = os.path.basename(f_path)
        try:
            # Read
            arr = np.loadtxt(f_path)
            if arr.ndim > 1:
                labels = arr[:, -1]
            else:
                labels = arr

            # Count
            total_points = len(labels)
            nan_count = np.isnan(labels).sum()

            if nan_count > 0:
                percent = (nan_count / total_points) * 100
                print(f"[WARN] {filename}")
                print(f"       Total Points: {total_points}")
                print(f"       NaN Points:   {nan_count} ({percent:.2f}%)")
                total_nans_all_files += nan_count
            else:
                pass

        except Exception as e:
            print(f"[ERR]  Could not read {filename}: {e}")

    print("\n" + "="*30)
    print(f"Total NaNs found across all files: {total_nans_all_files}")

    if total_nans_all_files > 0:
        print("\nHow to interpret the results:")
        print("1. If NaN % is < 5%:  Safe to DROP.")
        print("2. If NaN % is > 20%: The file is corrupt. Delete the whole file.")


if __name__ == "__main__":
    count_nans()
