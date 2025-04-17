import argparse
from pathlib import Path
import sys
import pandas as pd


parser = argparse.ArgumentParser(description="Extract 'context', 'b1-3' and 'observations' columns from given dataframes.")
parser.add_argument("infile", nargs=1, type=Path, help="the file containing the dataframe to extract data from")
args = parser.parse_args()

file, = args.infile

if not (file.exists() and file.is_file):
    print(f"error: '{file}' is not a file", file=sys.stderr)
    exit(1)

print("loading...")
all_data = pd.read_pickle(str(file.absolute()))

out_df = all_data[["context", "b1", "b2", "b3", "observations"]]

out_df.attrs = all_data.attrs

out_name = "observations-" + file.name
out_path = file.parent / out_name

print("pickling...")
pd.to_pickle(out_df, str(out_path))

print("done.")