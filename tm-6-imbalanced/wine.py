import polars as pl

df = pl.read_csv("wine.csv")

print(df.head())
print(f"Shape dataset: {df.shape}")
print(df.describe())

# melabel the target variable
df = df.with_columns(
    pl.when(pl.col("quality") >= 7).then(1).otherwise(0).alias("label")
).drop(["quality", "Id"])

# Check for missing values
null_count_df = df.null_count()
total_nulls = null_count_df.sum_horizontal().item()
print(null_count_df)
print(f"Total missing values: {total_nulls}")


# distribusi class
import matplotlib.pyplot as plt
import seaborn as sns

print("="*50)
print("Distribusi Kelas Sebelum Resampling")
print("="*50)

dist = df.group_by("label").len().sort("label")
print(dist)

plt.figure(figsize=(6, 4))
sns.barplot(x=dist["label"].to_list(), y=dist["len"].to_list(), palette="coolwarm")
plt.xticks(ticks=[0, 1], labels=["Bad Wine (0)", "Good Wine (1)"])
plt.xlabel("Kelas")
plt.ylabel("Jumlah")
plt.title("Distribusi Kelas Sebelum Resampling")
plt.savefig("before.png", dpi=150, bbox_inches="tight")
plt.show()

# konversi ke numpy untuk imblearn
import numpy as np

X = df.drop("label").to_numpy()
y = df["label"].to_numpy()

# Resampling menggunakan SMOTE
print("="*50)
print("Resampling dengan SMOTE")
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print("Distribusi sesudah SMOTE:", np.bincount(y_smote))

# Distribusi class sesudah SMOTE-ENN
print("="*50)
print("Distribusi Kelas Sesudah SMOTE_ENN")

from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=42)
X_enn, y_enn = smote_enn.fit_resample(X, y)

print("Distribusi sesudah SMOTE-ENN:", np.bincount(y_enn))