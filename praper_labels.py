import os
import json
import csv

TRAIN_FOLDER = os.path.join("data", "train_data", "train_data")
OUTPUT_CSV = os.path.join("data", "train_labels.csv")

rows = []

for filename in os.listdir(TRAIN_FOLDER):
    if filename.endswith(".json"):
        file_path = os.path.join(TRAIN_FOLDER, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_name = data["file_name"]
        direction = data["walking_into_frame_bool"]

        bbox = data["bbox"]

        top_left = bbox["top_left"]
        top_right = bbox["top_right"]
        bottom_left = bbox["bottom_left"]
        bottom_right = bbox["bottom_right"]

        x_values = [top_left[0], top_right[0], bottom_left[0], bottom_right[0]]
        y_values = [top_left[1], top_right[1], bottom_left[1], bottom_right[1]]

        xmin = min(x_values)
        xmax = max(x_values)
        ymin = min(y_values)
        ymax = max(y_values)

        rows.append([file_name, xmin, ymin, xmax, ymax, direction])

def extract_number(row):
    try:
        return int(row[0].split("_")[1])
    except (IndexError, ValueError):
        return float("inf")

rows.sort(key=extract_number)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "xmin", "ymin", "xmax", "ymax", "direction"])
    writer.writerows(rows)

print(f"Done. Saved {len(rows)} rows to {OUTPUT_CSV}")
print("First 5 rows:")
for row in rows[:5]:
    print(row)