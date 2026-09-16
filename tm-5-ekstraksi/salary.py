import polars as pl

df = pl.read_csv("salary_preprocessed.csv")

print(f"Shape dataset: {df.shape}")
print(df.describe())
print(df)

# hapus kolom yang tidak diperlukan
df = df.drop(["Salary_Category", "Job Title"])
print(f"\ndataset setelah menghapus kolom yang tidak diperlukan: {df.shape}")
print(df)


# n komponen bebas
print("\n" + "=" * 50)
print("PCA: Menentukan jumlah komponen bebas")
print("=" * 50)

# hot encode kolom kategorikal
df_encoded = df.to_dummies(columns=["Education Level", "Gender"])
print(f"\ndataset setelah hot encoding: {df_encoded.shape}")

# pisahkan fitur dan target
X = df_encoded.drop("Salary") #fitur
y = df_encoded["Salary"] #target
X = X.to_numpy() #fitur dalam bentuk numpy array

# standarisasi fitur numerik
# karena fitur numerik memiliki skala yang berbeda-beda, standarisasi diperlukan agar model machine learning dapat bekerja lebih baik.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (Principal Component Analysis) untuk reduksi dimensi
# PCA adalah teknik reduksi dimensi yang digunakan untuk mengurangi jumlah fitur dalam dataset sambil mempertahankan sebanyak mungkin informasi. 
from sklearn.decomposition import PCA

pca = PCA(n_components=2)          # n bebas → kita pilih 2 biar bisa divisualisasikan
X_pca = pca.fit_transform(X_scaled)

df_pca = pl.DataFrame(X_pca, schema=["PC1", "PC2"])
df_pca = df_pca.with_columns(y.alias("Salary"))   # tempel target balik

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", pca.explained_variance_ratio_.sum())


# Visualisasi hasil PCA
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sc = plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["Salary"],
                 cmap="viridis", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Salary Dataset (2 Komponen)")
plt.colorbar(sc, label="Salary")
plt.show()


# n komponen optimal 

print("\n" + "=" * 50)
print("PCA: Menentukan jumlah komponen optimal")
print("=" * 50)

import numpy as np

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