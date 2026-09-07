import polars as pl
from polars import selectors as cs

# baca dataset
print("=" * 50)
print("LOADING DATASET")
print("=" * 50)

df = pl.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\TUGAS\SEM V\Machine Learnig\Salary_Data.csv",
    schema_overrides={"Years of Experience": pl.Float64})
print(f"Shape dataset: {df.shape}")
print(f"5 baris pertama:", df.head(5))

# Cek MISSING VALUE
print("\n" + "=" * 50)
print("CEK MISSING VALUE")
print("=" * 50)

null_count_df = df.null_count()
# Sum semua kolom untuk mendapatkan total scalar
total_nulls = null_count_df.sum().row(0)[0]
print (null_count_df)

if total_nulls > 0:
    print("Terdapat Missing Values dalam dataset:")
    print(null_count_df)
    print(f"Total missing values: {total_nulls}")
else:
    print("Tidak terdapat Missing Values dalam dataset.")

# MENGGANTI MISSING VALUE DENGAN 3 METODE
print("\n" + "=" * 50)
print("HANDLING MISSING VALUE DENGAN 3 METODE")
print("=" * 50)

# 1. Menghapus missing values
print("\n1. DROP NULLS (Menghapus Missing Values)")
df_cleaned = df.drop_nulls()
print(f"Shape setelah menghapus nulls: {df_cleaned.shape}")
print(f"Jumlah baris yang dihapus: {df.shape[0] - df_cleaned.shape[0]}")

# 2. Mengganti dengan Mean
print("\n2. FILL WITH MEAN")
df_mean = df.with_columns(
    cs.numeric().fill_null(cs.numeric().mean())
)
print(f"Shape setelah fill dengan mean: {df_mean.shape}")
null_count_mean = df_mean.null_count().sum().row(0)[0]
print(f"Sisa missing values: {null_count_mean}")

# 3. Mengganti dengan Median
print("\n3. FILL WITH MEDIAN")
df_median = df.with_columns(
    cs.numeric().fill_null(cs.numeric().median())
)
print(f"Shape setelah fill dengan median: {df_median.shape}")
null_count_median = df_median.null_count().sum().row(0)[0]
print(f"Sisa missing values: {null_count_median}")

# 4. Mengganti dengan Modus
print("\n4. FILL WITH MODE")
# Untuk modus, kita perlu menghandle setiap kolom secara individual
numeric_cols = df.select(cs.numeric()).columns

df_mode = df.clone()
for col in numeric_cols:
    if df[col].null_count() > 0:
        # Get mode (most common value)
        mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
        df_mode = df_mode.with_columns(
            pl.col(col).fill_null(mode_value)
        )

print(f"Shape setelah fill dengan mode: {df_mode.shape}")
null_count_mode = df_mode.null_count().sum().row(0)[0]
print(f"Sisa missing values: {null_count_mode}")

# PERBANDINGAN KETIGA METODE
print("\n" + "=" * 50)
print("PERBANDINGAN KETIGA METODE")
print("=" * 50)

comparison = pl.DataFrame({
    "Metode": ["Drop Nulls", "Fill Mean", "Fill Median", "Fill Mode"],
    "Total Rows": [df_cleaned.shape[0], df_mean.shape[0], df_median.shape[0], df_mode.shape[0]],
    "Rows Removed/Kept": [
        df.shape[0] - df_cleaned.shape[0],
        0,
        0,
        0
    ],
    "Missing Values Sisa": [0, null_count_mean, null_count_median, null_count_mode]
})

print(comparison)

# OPSI: Simpan hasil ke CSV
print("\n" + "=" * 50)
print("SIMULASI METODE HANDLING MISSING VALUE")
print("=" * 50)
print("✓ Simulasi selesai - File tidak disimpan (hanya simulasi)")

print("\nMelanjutkan ke Outlier Handling dengan metode MODUS...")

# OUTLIER HANDLING DENGAN IQR CAPPING
print("\n" + "=" * 50)
print("CEK & HANDLE OUTLIER (IQR CAPPING)")
print("=" * 50)

# Gunakan df_mode sebagai baseline (metode fill mode)
df_outlier = df_mode.clone()

# Ambil kolom numerik saja
numeric_cols_for_outlier = df_outlier.select(cs.numeric()).columns

# Deteksi Outlier Sebelum Handling
print("\n1. DETEKSI OUTLIER SEBELUM HANDLING")
total_outliers_before = {}
for col in numeric_cols_for_outlier:
    q1 = df_outlier[col].quantile(0.25)
    q3 = df_outlier[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Hitung jumlah baris outlier
    count = df_outlier.filter(
        (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
    ).height
    total_outliers_before[col] = count
    
    if count > 0:
        print(f"Kolom '{col}': {count} outlier (Rentang Normal: {lower_bound:.2f} s/d {upper_bound:.2f})")

total_outliers_count = sum(total_outliers_before.values())
print(f"\nTotal outliers di semua kolom: {total_outliers_count}")

# Capping Outlier Menggunakan .clip()
print("\n2. CAPPING OUTLIER")
exprs_cap = []
for col in numeric_cols_for_outlier:
    q1 = df_outlier[col].quantile(0.25)
    q3 = df_outlier[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    exprs_cap.append(
        pl.col(col).clip(lower_bound=lower_bound, upper_bound=upper_bound)
    )

# Jalankan eksekusi transformasi sekaligus
df_no_outliers = df_outlier.with_columns(exprs_cap)

print(f"Shape dataset setelah capping outlier: {df_no_outliers.shape}")

# Perbandingan Statistik Sebelum & Sesudah Capping
print("\n3. PERBANDINGAN STATISTIK SEBELUM & SESUDAH CAPPING")
for col in numeric_cols_for_outlier:
    if total_outliers_before[col] > 0:
        print(f"\nKolom: {col}")
        print(f"  Sebelum - Min: {df_outlier[col].min():.2f}, Max: {df_outlier[col].max():.2f}, Mean: {df_outlier[col].mean():.2f}")
        print(f"  Sesudah - Min: {df_no_outliers[col].min():.2f}, Max: {df_no_outliers[col].max():.2f}, Mean: {df_no_outliers[col].mean():.2f}")

# Simpan hasil akhir
print("\n" + "=" * 50)
print("SIMPAN HASIL AKHIR PREPROCESSING")
print("=" * 50)

df_no_outliers.write_csv("Salary_Data_Preprocessed_Mode_IQR.csv")
# print("✓ Salary_Data_Preprocessed_Mode_IQR.csv")

print("\n" + "=" * 50)
print("RINGKASAN AKHIR")
print("=" * 50)
print(f"Shape awal dataset: {df.shape}")
print(f"Metode Missing Value: MODE (Modus)")
print(f"Shape dataset setelah handling missing values: {df_mode.shape}")
print(f"Shape dataset setelah handling outliers (IQR Capping): {df_no_outliers.shape}")
print(f"Total outliers yang berhasil di-cap: {total_outliers_count}")
print("\nPreprocessing LENGKAP! ✓")
