import sys
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

sys.stdout.reconfigure(encoding="utf-8")

df = pl.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\TUGAS\SEM V\Machine Learnig\WineQT.csv")

print("=" * 60)
print("STEP 1 - LOAD DATASET")
print("=" * 60)
print(df.head())
print(f"\nShape dataset: {df.shape}")
print(df.describe())

# label biner: quality >= 7 -> 1 (good), selainnya 0 (bad)
df = df.with_columns(
    pl.when(pl.col("quality") >= 7).then(1).otherwise(0).alias("label")
).drop(["quality", "Id"])

print("\n" + "=" * 60)
print("STEP 2 - LABEL BINER (quality >= 7 -> 1, selainnya -> 0)")
print("=" * 60)
print(df.head())

#  cek missing value
print("\n" + "=" * 60)
print("STEP 3 - CEK MISSING VALUE")
print("=" * 60)
print(df.null_count())
total_nulls = df.null_count().sum_horizontal().item()
print(f"Total missing values: {total_nulls}")
if total_nulls == 0:
    print("Tidak terdapat Missing Values dalam dataset.")

#  distribusi kelas sebelum resampling
print("\n" + "=" * 60)
print("STEP 4 - DATA & DISTRIBUSI KELAS SEBELUM RESAMPLING")
print("=" * 60)
print(f"Shape data sebelum resampling: {df.shape}")
print("\nData sebelum resampling (10 baris pertama):")
print(df.head(10))
print("\nDistribusi kelas:")
dist = df.group_by("label").len().sort("label")
print(dist)


def plot_distribusi(counts, judul, nama_file):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=[0, 1], y=counts, palette="coolwarm", hue=[0, 1], legend=False)
    plt.xticks(ticks=[0, 1], labels=["Bad Wine (0)", "Good Wine (1)"])
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah")
    plt.title(judul)
    plt.savefig(nama_file, dpi=150, bbox_inches="tight")
    plt.show()


sebelum = [dist.filter(pl.col("label") == 0)["len"][0], dist.filter(pl.col("label") == 1)["len"][0]]
print(f"Bad Wine (0): {sebelum[0]} baris")
print(f"Good Wine (1): {sebelum[1]} baris")
plot_distribusi(sebelum, "Distribusi Kelas Sebelum Resampling", "before.png")

# konversi ke numpy untuk imblearn
X = df.drop("label").to_numpy()
y = df["label"].to_numpy()

# 4 metode resampling
metode = {
    "SMOTE": SMOTE(random_state=42),
    "SMOTE-ENN": SMOTEENN(random_state=42),
    "ROS": RandomOverSampler(random_state=42),
    "RUS": RandomUnderSampler(random_state=42),
}

hasil = {}
for nama, sampler in metode.items():
    print("\n" + "=" * 60)
    print(f"STEP 6 - {nama}")
    print("=" * 60)
    X_res, y_res = sampler.fit_resample(X, y)
    counts = [int(c) for c in np.bincount(y_res)]
    hasil[nama] = counts
    print(f"Distribusi sebelum {nama}: Bad={sebelum[0]}, Good={sebelum[1]}")
    print(f"Distribusi sesudah {nama}: Bad={counts[0]}, Good={counts[1]}")
    print(f"Jumlah data sesudah {nama}: {len(y_res)} baris")
    plot_distribusi(counts, f"Distribusi Kelas Sesudah {nama}", f"{nama.lower().replace('-', '')}_after.png")

# ringkasan perbandingan
print("\n" + "=" * 60)
print("RINGKASAN PERBANDINGAN")
print("=" * 60)
ringkasan = pl.DataFrame({
    "Metode": ["Sebelum"] + list(hasil.keys()),
    "Kelas 0 (bad)": [sebelum[0]] + [hasil[m][0] for m in hasil],
    "Kelas 1 (good)": [sebelum[1]] + [hasil[m][1] for m in hasil],
    "Total": [sum(sebelum)] + [sum(hasil[m]) for m in hasil],
})
print(ringkasan)
