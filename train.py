import kagglehub
import pandas as pd
from pathlib import Path
import shutil

script_directory = Path(__file__).parent.resolve()
destination_subfolder_name = "dataset"
final_data_dir = script_directory / destination_subfolder_name

final_data_dir.mkdir(parents=True, exist_ok=True)

dataset_cache_root_path: Path = Path(kagglehub.dataset_download("siddharth0935/himalayan-expeditions"))

actual_csv_file_name = "exped.csv"
source_csv_file_path = dataset_cache_root_path / actual_csv_file_name

if source_csv_file_path.exists():
    # Define the target path for the file in your project folder ---
    target_csv_file_path = final_data_dir / actual_csv_file_name

    # Copy the file from the cache to your project folder ---
    try:
        shutil.copy(source_csv_file_path, target_csv_file_path)

        # Now, load the CSV file from your project folder
        df = pd.read_csv(target_csv_file_path)

    except Exception as e:
        print(f"An error occurred during file copy or read: {e}")