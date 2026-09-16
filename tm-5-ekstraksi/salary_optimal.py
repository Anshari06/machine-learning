import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pl.read_csv("salary_preprocessed.csv")

# hapus kolom yang tidak diperlukan
df = df.drop(["Salary_Category", "Job Title"])

# hot encode kolom kategorikal
df_encoded = df.to_dummies(columns=["Education Level", "Gender"])

# pisahkan fitur dan target
X = df_encoded.drop("Salary") #fitur
y = df_encoded["Salary"] #target
X = X.to_numpy() #fitur dalam bentuk numpy array

# standarisasi fitur numerik
# karena fitur numerik memiliki skala yang berbeda-beda, standarisasi diperlukan agar PCA tidak didominasi fitur berskala besar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA: Menentukan jumlah komponen optimal
print("=" * 50)
print("PCA: Menentukan jumlah komponen optimal")
print("=" * 50)

pca_full = PCA(n_components=None)   # hitung SEMUA komponen
X_full = pca_full.fit_transform(X_scaled)

ratio = pca_full.explained_variance_ratio_
cum = np.cumsum(ratio)                          # varians kumulatif

for i, (r, c) in enumerate(zip(ratio, cum), start=1):
    print(f"PC{i:2d}: {r:.4f}  kumulatif: {c:.4f}")

n_optimal = np.argmax(cum >= 0.90) + 1          # komponen PERTAMA yang kumulatifnya >= 90%
print(f"\nn optimal = {n_optimal} komponen (kumulatif {cum[n_optimal-1]:.2%})")

# scree plot untuk menentukan jumlah komponen optimal
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ratio) + 1), ratio, marker="o", label="Per komponen")
plt.plot(range(1, len(cum) + 1), cum, marker="s", label="Kumulatif")
plt.axhline(0.90, color="red", linestyle="--", label="Ambang 90%")
plt.axvline(n_optimal, color="gray", linestyle=":", label=f"n optimal = {n_optimal}")
plt.xlabel("Komponen Utama"); plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot — Pemilihan n Komponen Optimal")
plt.legend(); plt.show()

# transformasi ulang dataset dengan n komponen optimal
pca_opt = PCA(n_components=n_optimal)
X_opt = pca_opt.fit_transform(X_scaled)

df_opt = pl.DataFrame(X_opt, schema=[f"PC{i}" for i in range(1, n_optimal + 1)])
df_opt = df_opt.with_columns(y.alias("Salary"))

# simpan hasil ekstraksi fitur (n komponen optimal)
df_opt.write_csv("salary_pca_optimal.csv")
print(f"\nHasil disimpan ke salary_pca_optimal.csv ({df_opt.shape[1] - 1} komponen + Salary)")
