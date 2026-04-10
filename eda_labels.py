import pandas as pd

df = pd.read_csv("data/train_labels.csv")

df["width"] = df["xmax"] - df["xmin"]
df["height"] = df["ymax"] - df["ymin"]
df["center_x"] = (df["xmin"] + df["xmax"]) / 2
df["center_y"] = (df["ymin"] + df["ymax"]) / 2

print("Total samples:", len(df))

print("\nDirection counts:")
print(df["direction"].value_counts())

print("\nCoordinate summary:")
print(df[["xmin", "ymin", "xmax", "ymax"]].describe())

print("\nBox size summary:")
print(df[["width", "height"]].describe())

print("\nCenter summary:")
print(df[["center_x", "center_y"]].describe())

print("\nHow many boxes fully left of frame (xmax < 0):", (df["xmax"] < 0).sum())
print("How many boxes fully right of frame (xmin > 720):", (df["xmin"] > 720).sum())
print("How many boxes at least partially inside frame:", ((df["xmax"] >= 0) & (df["xmin"] <= 720)).sum())