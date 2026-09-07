import polars as pl
import io
import sys
import polars.selectors as cs

# Fix Unicode encoding untuk Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 1. Loading Dataset
print("=" * 50)
print("LOADING DATASET")
print("=" * 50)
df = pl.read_csv("WineQT.csv")
print(f"Shape dataset: {df.shape}")
print(f"5 baris pertama:\n{df.head(5)}")

# Cek Data Numerical vs Categorical
print("\n" + "=" * 50)
print("CEK DATA NUMERICAL VS CATEGORICAL")
print("=" * 50)


numeric_cols = df.select(cs.numeric()).columns
categorical_cols = df.select(cs.string() | cs.categorical()).columns

df_numeric = df.select(numeric_cols)
df_categorical = df.select(categorical_cols)

# Buat tabel klasifikasi tipe data tiap kolom
tabel_tipe = pl.DataFrame({
    "Nama Kolom": df.columns,
    "Tipe Data": [str(df[col].dtype) for col in df.columns],
    "Kategori": [
        "Numerical" if col in numeric_cols else "Categorical"
        for col in df.columns
    ],
})
print(tabel_tipe)
print(f"\nTotal kolom numerical  : {len(numeric_cols)}")
if len(categorical_cols) > 0:
    print(f"Total kolom categorical: {len(categorical_cols)}")
else:
    print("Total kolom categorical: 0 (Tidak ada kolom categorical dalam dataset ini.)")

# 2. Cek Missing Value
print("\n" + "=" * 50)
print("CEK MISSING VALUE")
print("=" * 50)
null_counts = df.null_count()
total_nulls = sum(null_counts.row(0))  # Menjumlahkan null di seluruh kolom

if total_nulls > 0:
  print("Terdapat Missing Values dalam dataset:")
  print(null_counts)
  print(f"Total missing values: {total_nulls}")
else:
  print("Tidak terdapat Missing Values dalam dataset.")

# 3. Handling Missing Values
print("\n" + "=" * 50)
print("HANDLING MISSING VALUE (PERSIAPAN)")
print("=" * 50)

df_cleaned = df.drop_nulls()

df_mean = df.with_columns(
    pl.col(pl.NUMERIC_DTYPES).fill_null(pl.col(pl.NUMERIC_DTYPES).mean())
)

df_median = df.with_columns(
    pl.col(pl.NUMERIC_DTYPES).fill_null(pl.col(pl.NUMERIC_DTYPES).median())
)

numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
mode_exprs = [
    pl.col(col).fill_null(
        df[col].mode()[0] if len(df[col].mode()) > 0 else 0
    )
    for col in numeric_cols
]
df_mode = df.with_columns(mode_exprs)

# 4. Outlier Handling dengan IQR Capping
print("\n" + "=" * 50)
print("CEK & HANDLE OUTLIER (IQR CAPPING)")
print("=" * 50)

# Pilih kolom fitur kimiawi saja (abaikan Id dan target quality)
fitur_kimia = [col for col in numeric_cols if col not in ["Id", "quality"]]

# Deteksi Outlier Sebelum Handling
total_outliers_before = {}
for col in fitur_kimia:
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

total_outliers_count = sum(total_outliers_before.values())
print(f"\nTotal outliers di semua fitur kimiawi: {total_outliers_count}")

# Capping Outlier Menggunakan .clip()
exprs_cap = []
for col in fitur_kimia:
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)
  iqr = q3 - q1
  lower_bound = q1 - 1.5 * iqr
  upper_bound = q3 + 1.5 * iqr

  exprs_cap.append(
      pl.col(col).clip(lower_bound=lower_bound, upper_bound=upper_bound)
  )

# Jalankan eksekusi transformasi sekaligus
df_no_outliers = df.with_columns(exprs_cap)

print(f"\nShape dataset setelah capping outlier: {df_no_outliers.shape}")

# Perbandingan Statistik Sebelum & Sesudah Capping
print("\n3. PERBANDINGAN STATISTIK SEBELUM & SESUDAH CAPPING")
for col in fitur_kimia:
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

print("\n" + "=" * 50)
print("RINGKASAN AKHIR")
print("=" * 50)
print(f"Shape awal dataset: {df.shape}")
print(f"Shape dataset bersih: {df_no_outliers.shape}")
print(f"Total titik nilai outlier yang berhasil di-cap: {total_outliers_count}")