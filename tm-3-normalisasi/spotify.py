import polars as pl
from polars import selectors as cs

df = pl.read_csv(
    r"C:\Users\LENOVO\OneDrive\Documents\TUGAS\SEM V\Machine Learnig\songs_normalize.csv"
).unique()
print(f"Shape dataset: {df.shape}")
print(f"5 baris pertama:\n{df.head(5)}")


# cek ulang apakah ada missing value
print("\n" + "=" * 50)
print("CEK MISSING VALUE")
print("=" * 50)

if df.null_count().sum_horizontal().item() > 0:
    print("Terdapat Missing Values dalam dataset:")
    print(df.null_count())
    print(f"Total missing values: {df.null_count().sum_horizontal().item()}")
else:
    print("Tidak terdapat Missing Values dalam dataset.")

# cek deskripsi data sebelum normalisasi
print("\n----" + "cek deskripsi data sebelum normalisasi" + "----")
print("cek deskripsi data sebelum normalisasi:\n", df.describe())

# cek data numerical vs categorical
print("\n" + "=" * 50)
print("CEK DATA NUMERICAL VS CATEGORICAL")
print("=" * 50)

categorical_cols = df.select(cs.string() | cs.categorical()).columns
numeric_cols = df.select(cs.numeric()).columns

print(f"Kolom Kategorikal: {categorical_cols}")
print(f"Total kolom kategorikal: {len(categorical_cols)}")

print(f"\nKolom Numerik: {numeric_cols}")
print(f"Total kolom numerik: {len(numeric_cols)}")

# CEK OUTLIER DENGAN METODE IQR
print("\n" + "=" * 50)
print("CEK OUTLIER DENGAN METODE IQR")
print("=" * 50)

# Kita saring: Lewatkan 'year', 'key', dan 'mode' karena itu data kategori angka, tidak punya outlier.
fitur_musik = [col for col in numeric_cols if col not in ["year", "key", "mode"]]

total_outliers_before = {}
iqr_bounds = {}
for column in fitur_musik:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # simpan batas IQR untuk dipakai ulang di capping dan verifikasi
    iqr_bounds[column] = (lower_bound, upper_bound)

    outlier_count = df.filter(
        (pl.col(column) < lower_bound) |
        (pl.col(column) > upper_bound)
    ).height
    total_outliers_before[column] = outlier_count
    print(f"Kolom '{column}': ditemukan {outlier_count} outlier")

print(f"\nTotal outliers di semua fitur musik: {sum(total_outliers_before.values())}")

# OUTLIER HANDLING (IQR CAPPING)
print("\n" + "=" * 50)
print("PROSES IQR CAPPING")
print("=" * 50)

exprs_cap = []
for column in fitur_musik:
    lower_bound, upper_bound = iqr_bounds[column]

    # cast ke Float64: clip pada kolom Int64 memotong batas desimal (mis. 30.5 -> 30),
    # sehingga nilai hasil capping masih di bawah batas dan outlier tak pernah habis
    exprs_cap.append(
        pl.col(column).cast(pl.Float64).clip(lower_bound=lower_bound, upper_bound=upper_bound)
    )

# with_columns tidak mengubah df asli (polars immutable), hasilnya langsung jadi df_clean
# dengan semua kolom tetap ada, hanya 11 fitur musik yang di-cap
df_clean = df.with_columns(exprs_cap)

print("[INFO] IQR Capping selesai.")
print(f"Shape data setelah capping: {df_clean.shape}")

# Verifikasi: cek sisa outlier terhadap batas yang sama dengan yang dipakai capping
print("\n" + "=" * 50)
print("VERIFIKASI SETELAH IQR CAPPING")
print("=" * 50)

sisa_outlier = 0
for column in fitur_musik:
    lower_bound, upper_bound = iqr_bounds[column]

    count = df_clean.filter(
        (pl.col(column) < lower_bound) |
        (pl.col(column) > upper_bound)
    ).height
    sisa_outlier += count
    print(f"{column}: {count} outlier")

print(f"\nTotal sisa outlier: {sisa_outlier}")

# Perbandingan statistik sebelum & sesudah capping (hanya kolom yang punya outlier)
print("\nPerbandingan statistik sebelum & sesudah capping:")

comparison_data = []
for column in fitur_musik:
    if total_outliers_before[column] > 0:
        comparison_data.append({
            "Kolom": column,
            "Outliers": total_outliers_before[column],
            "Min Sebelum": df[column].min(),
            "Max Sebelum": df[column].max(),
            "Mean Sebelum": df[column].mean(),
            "Min Sesudah": df_clean[column].min(),
            "Max Sesudah": df_clean[column].max(),
            "Mean Sesudah": df_clean[column].mean()
        })

if comparison_data:
    comparison_df = pl.DataFrame(comparison_data)
    print(comparison_df)

# mulai menormalisasi data numerik
print("\n" + "=" * 50)
print("NORMALISASI DATA NUMERIK")
print("=" * 50)

# metode 1 Simple Feature Scaling
# melakukan scaling data menggunakan nilai maksimum sehingga nilai hasil transformasi berada dalam rentang yang lebih kecil.
# contoh 7,10,15,25; maka nilain maksimum diambil, kemudian semua nilai dibagi dengan 25, sehingga hasil transformasi menjadi 0.28, 0.4, 0.6, 1.0
print("\n SIMPLE FEATURE SCALING")
simple_scaling = df_clean.with_columns([
    (pl.col(col) / pl.col(col).max()).alias(col)
    for col in fitur_musik
])

print("\nHasil Simple Feature Scaling:")
print(simple_scaling.select(fitur_musik).head(5))

# metode 2 Min-Max Scaling
# melakukan scaling data menggunakan nilai minimum dan maksimum sehingga nilai hasil transformasi berada dalam rentang [0, 1].
print("\n MIN-MAX SCALING")
min_max_scaling = df_clean.with_columns([
    ((pl.col(col) - pl.col(col).min())
     /
    (pl.col(col).max() - pl.col(col).min())).alias(col)
    for col in fitur_musik
])

print("\nHasil Min-Max Scaling:")
print(min_max_scaling.select(fitur_musik).head(5))

# metode 3 Z-Score Normalization
# melakukan normalisasi data menggunakan rata-rata dan standar deviasi sehingga nilai hasil transformasi memiliki distribusi normal dengan rata-rata 0 dan standar deviasi 1.
print("\n Z-SCORE NORMALIZATION")
z_score_normalization = df_clean.with_columns([
    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
    for col in fitur_musik
])

print("\nHasil Z-Score Normalization:")
print(z_score_normalization.select(fitur_musik).head(5))

# contoh perbandingan hasil normalisasi pada kolom tempo
print("\nPerbandingan hasil normalisasi pada kolom 'tempo':")
print(f"\nSebelum Normalisasi (setelah capping outlier):\n{df_clean.select('tempo').head(5)}")
print(f"\nSimple Feature Scaling:\n{simple_scaling.select('tempo').head(5)}")
print(f"\nMin-Max Scaling:\n{min_max_scaling.select('tempo').head(5)}")
print(f"\nZ-Score Normalization:\n{z_score_normalization.select('tempo').head(5)}")
