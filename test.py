import kagglehub
import pandas as pd
from pathlib import Path
import shutil

script_directory = Path(__file__).parent.resolve()
destination_subfolder_name = "dataset"
final_data_dir = script_directory / destination_subfolder_name

final_data_dir.mkdir(parents=True, exist_ok=True)
print(f"Data will be copied to: {final_data_dir}")

print("Downloading dataset to KaggleHub cache...")
dataset_cache_root_path: Path = Path(kagglehub.dataset_download("siddharth0935/himalayan-expeditions"))
print(f"Dataset downloaded to KaggleHub cache at: {dataset_cache_root_path}")

actual_csv_file_name = "exped.csv"
source_csv_file_path = dataset_cache_root_path / actual_csv_file_name
print(f"Source CSV file in cache: {source_csv_file_path}")

if source_csv_file_path.exists():
    print(f"CONFIRMED: CSV file found at {source_csv_file_path}")

    # Define the target path for the file in your project folder ---
    target_csv_file_path = final_data_dir / actual_csv_file_name
    print(f"Target path for copied CSV: {target_csv_file_path}")

    # Copy the file from the cache to your project folder ---
    try:
        print(f"Copying '{source_csv_file_path.name}' to '{final_data_dir}'...")
        shutil.copy(source_csv_file_path, target_csv_file_path)
        print(f"File copied to: {target_csv_file_path}")

        # Now, load the CSV file from your project folder
        df = pd.read_csv(target_csv_file_path)
        print("\nFirst 5 records from copied file:")
        print(df.head())

    except Exception as e:
        print(f"An error occurred during file copy or read: {e}")
