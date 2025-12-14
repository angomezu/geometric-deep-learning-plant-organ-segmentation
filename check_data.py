import os
import open3d as o3d
import numpy as np
import glob

# Path to your raw data
DATA_DIR = r"C:\Oak Ridge\Model\data\raw"


def scan_files():
    # We need to get all .pcd files
    pcd_files = glob.glob(os.path.join(DATA_DIR, "*_filtered.pcd"))

    if not pcd_files:
        print("ERROR: No .pcd files found! Check your path.")
        return

    bad_files = []

    for f_path in pcd_files:
        filename = os.path.basename(f_path)
        try:
            # Attempt to read
            pcd = o3d.io.read_point_cloud(f_path)
            num_points = len(pcd.points)

            if num_points == 0:
                print(f"[FAIL] {filename}: 0 points (CORRUPTED/EMPTY)")
                bad_files.append(filename)
            else:
                print(f"[OK]   {filename}: {num_points} points")

        except Exception as e:
            print(f"[ERR]  {filename}: Read error - {e}")
            bad_files.append(filename)

    print("\n" + "="*30)
    if len(bad_files) == 0:
        print("All files look healthy.")
    elif len(bad_files) == len(pcd_files):
        print("CRITICAL: ALL files are reading as empty.")
        print("Diagnosis: Incompatibility between CloudCompare export and Open3D.")
        print("Solution: Re-export as .PLY in CloudCompare.")
    else:
        print(f"Found {len(bad_files)} bad files:")
        for b in bad_files:
            print(f"  - {b}")
        print("Solution: Re-export only these specific files.")


if __name__ == "__main__":
    scan_files()
