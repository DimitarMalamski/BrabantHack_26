import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "data/train_labels.csv"
OUTPUT_CSV = "data/split.csv"

df = pd.read_csv(INPUT_CSV)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["direction"]
)

train_ids = set(train_df["id"])
val_ids = set(val_df["id"])

split_rows = []

for image_id in df["id"]:
    if image_id in train_ids:
        split_rows.append([image_id, "train"])
    else:
        split_rows.append([image_id, "val"])

split_df = pd.DataFrame(split_rows, columns=["id", "split"])
split_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved split file to {OUTPUT_CSV}")
print(split_df["split"].value_counts())
print("\nDirection distribution in full data:")
print(df["direction"].value_counts())
print("\nDirection distribution in train:")
print(train_df["direction"].value_counts())
print("\nDirection distribution in val:")
print(val_df["direction"].value_counts())