
import os

def rename_csv_files(folder_path):
    # Loop through all files in the given folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .csv and starts with 'i7'
        if filename.startswith("i7"):
            # Create the new filename by replacing 'i7' with 'ref'
            new_filename = "ref" + filename[2:]
            # Get full paths
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

# Example usage:
rename_csv_files(r"C:\work\0_currently_in_use\1.0 - Copy\5_results\6_other_estimators\2_results\lon")
