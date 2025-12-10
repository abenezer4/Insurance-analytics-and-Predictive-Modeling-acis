import os
import sys
from pathlib import Path

# ---------------------------------------------------------
# Ensure that src/ folder is in Python path
# So we can import src.data_loader and src.cleaning
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))   # src/
project_root = os.path.abspath(os.path.join(current_dir, ".."))  # project root
src_path = os.path.join(project_root, "src")               # project_root/src

sys.path.append(src_path)

# ---------------------------------------------------------
# Now imports will work (even under dvc repro)
# ---------------------------------------------------------
from data_loader import load_data
from cleaning import clean_data


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
DATA_DIR = os.path.join(project_root, "data")
RAW_FILE = os.path.join(DATA_DIR, "MachineLearningRating_v3.txt")
CLEAN_FILE = os.path.join(DATA_DIR, "clean_data.csv")

# Adjust delimiter here if needed
DELIMITER = "|"


def main():
    print("Working directory:", os.getcwd())
    print("Loading raw dataset from:", RAW_FILE)

    df = load_data("MachineLearningRating_v3.txt", sep=DELIMITER)

    print("Cleaning dataset...")
    df_clean = clean_data(df)

    print("Saving cleaned dataset to:", CLEAN_FILE)
    df_clean.to_csv(CLEAN_FILE, index=False)

    print("\n✔ Cleaning step complete.")


if __name__ == "__main__":
    main()
