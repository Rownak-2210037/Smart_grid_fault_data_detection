import pandas as pd
import numpy as np
import glob
import os


DATA_FOLDER = "triple/"           
OUTPUT_FILE = "output/ornl_merged.csv"


os.makedirs("output", exist_ok=True)


all_files = sorted(glob.glob(DATA_FOLDER + "*.csv"))
print(f" files: {len(all_files)}")

if len(all_files) == 0:
    print("❌ ERROR:in data/ folder no csv")
    print("  keep 15 file in data/ folder ")
    exit()


df_list = []
for i, filepath in enumerate(all_files, 1):
    df_temp = pd.read_csv(filepath)
    df_list.append(df_temp)
    print(f"  ✅ File {i:02d}: {os.path.basename(filepath)}"
          f" — {df_temp.shape[0]} rows"
          f" | classes: {df_temp['marker'].value_counts().to_dict()}")


df_merged = pd.concat(df_list, ignore_index=True)

print(f"\n📊 Merged shape: {df_merged.shape}")
print(f"📊 Class distribution:\n{df_merged['marker'].value_counts()}")

df_merged.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved: {OUTPUT_FILE}")
