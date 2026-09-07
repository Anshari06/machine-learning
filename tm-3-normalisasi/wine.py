import polars as pl
from polars import selectors as cs

df = pl.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\TUGAS\SEM V\Machine Learnig\WineQT.csv")
print(f"Shape dataset: {df.shape}")
print(f"5 baris pertama:\n{df.head(5)}")

# print schema
print("\n" + "=" * 50)
print("SCHEMA DATASET")
print("=" * 50)
print(df.schema)

# Cek Data Numerical vs Categorical
print("\n" + "=" * 50)
print("CEK DATA NUMERICAL VS CATEGORICAL")
print("=" * 50)
categorical_cols = df.select(cs.string() | cs.categorical()).columns
print(f"Kolom Kategorikal: {categorical_cols}")
print(f"Total kolom kategorikal: {len(categorical_cols)}")

numeric_cols = df.select(cs.numeric()).columns
print(f"Kolom Numerik: {numeric_cols}")
print(f"Total kolom numerik: {len(numeric_cols)}")

# Memisahkan kolom Id dan Quality dari data
print("\n" + "=" * 50)
print("MEMISAHKAN KOLOM ID DAN QUALITY")
print("=" * 50)

id_quality = df.select(['Id', 'quality'])
data_to_normalize = df.drop(['Id', 'quality'])

print(f"Data Id dan Quality:\n{id_quality.head()}")
print(f"\nShape Id & Quality: {id_quality.shape}")
print(f"\nKolom yang akan dinormalisasi: {data_to_normalize.columns}")
print(f"Shape data untuk normalisasi: {data_to_normalize.shape}")

print(f"\nDeskripsi data sebelum normalisasi:\n{data_to_normalize.describe()}")

# Handling outlier dengan IQR Capping (dari TM-2)
# outlier dideteksi dengan metode IQR: nilai di luar [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
# kemudian di-cap (dipotong) ke batas rentang normal, sehingga baris data tidak dibuang
print("\n" + "=" * 50)
print("OUTLIER HANDLING (IQR CAPPING)")
print("=" * 50)

total_outliers_before = {}
exprs_cap = []
for col in data_to_normalize.columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Hitung jumlah baris outlier
    count = df.filter(
        (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
    ).height
    total_outliers_before[col] = count
    if count > 0:
        print(
            f"Kolom '{col}': {count} outlier (Rentang Normal: {lower_bound:.2f} s/d"
            f" {upper_bound:.2f})"
        )

    exprs_cap.append(
        pl.col(col).clip(lower_bound=lower_bound, upper_bound=upper_bound)
    )

df_no_outliers = df.with_columns(exprs_cap)
total_outliers_count = sum(total_outliers_before.values())
print(f"\nTotal outliers di semua fitur: {total_outliers_count}")
print(f"Shape dataset setelah capping outlier: {df_no_outliers.shape}")

# Perbandingan statistik sebelum & sesudah capping
print("\nPerbandingan statistik sebelum & sesudah capping:")
for col in data_to_normalize.columns:
    if total_outliers_before[col] > 0:
        print(f"\nKolom: {col}")
        print(
            f"  Sebelum - Min: {df[col].min():.2f}, Max: {df[col].max():.2f}, Mean:"
            f" {df[col].mean():.2f}"
        )
        print(
            f"  Sesudah - Min: {df_no_outliers[col].min():.2f}, Max:"
            f" {df_no_outliers[col].max():.2f}, Mean:"
            f" {df_no_outliers[col].mean():.2f}"
        )

# mulai menormalisasi data numerik (menggunakan data yang sudah bebas outlier)
print("\n" + "=" * 50)
print("NORMALISASI DATA NUMERIK")
print("=" * 50)

# metode 1 Simple Feature Scaling
# melakukan scaling data menggunakan nilai maksimum sehingga nilai hasil transformasi berada dalam rentang yang lebih kecil.
# contoh 7,10,15,25; maka nilain maksimum diambil, kemudian semua nilai dibagi dengan 25, sehingga hasil transformasi menjadi 0.28, 0.4, 0.6, 1.0

print("\n SIMPLE FEATURE SCALING")
simple_scaling = df_no_outliers.with_columns([
    (pl.col(col) / pl.col(col).max()).alias(col)
    for col in data_to_normalize.columns
])

print("\nHasil Simple Feature Scaling:")
print(simple_scaling.head())

# metode 2 Min-Max Scaling
# melakukan scaling data menggunakan nilai minimum dan maksimum sehingga nilai hasil transformasi berada dalam rentang [0, 1].
print("\n MIN-MAX SCALING")
min_max_scaling = df_no_outliers.with_columns([
    ((pl.col(col) - pl.col(col).min()) 
     /
    (pl.col(col).max() - pl.col(col).min())).alias(col)
    for col in data_to_normalize.columns
])

print("\nHasil Min-Max Scaling:")
print(min_max_scaling.head())

# metode 3 Z-Score Normalization
# melakukan normalisasi data menggunakan rata-rata dan standar deviasi sehingga nilai hasil transformasi memiliki distribusi normal dengan rata-rata 0 dan standar deviasi 1.
print("\n Z-SCORE NORMALIZATION")
z_score_normalization = df_no_outliers.with_columns([
    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
    for col in data_to_normalize.columns
])
print("\nHasil Z-Score Normalization:")
print(z_score_normalization.head())


# contoh perbandingan hasil normalisasi pada kolom alcohol
print("\nPerbandingan hasil normalisasi pada kolom 'alcohol':")
print(f"\nSebelum Normalisasi (setelah capping outlier):\n{df_no_outliers.select('alcohol').head(5)}")
print(f"\nSimple Feature Scaling:\n{simple_scaling.select('alcohol').head(5)}")
print(f"\nMin-Max Scaling:\n{min_max_scaling.select('alcohol').head(5)}")
print(f"\nZ-Score Normalization:\n{z_score_normalization.select('alcohol').head(5)}")