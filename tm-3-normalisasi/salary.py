import polars as pl
from polars import selectors as cs

df = pl.read_csv(
    r"C:\Users\LENOVO\OneDrive\Documents\TUGAS\SEM V\Machine Learnig\Salary_Data.csv",
    schema_overrides={"Years of Experience": pl.Float64}
)

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


# cek missing value
print("\n" + "=" * 50)
print("CEK MISSING VALUE")
print("=" * 50)

null_count_df = df.null_count()
# Sum horizontal semua kolom untuk mendapatkan total scalar
total_nulls = null_count_df.sum_horizontal().item()
print (null_count_df)

if total_nulls > 0:
    print("Terdapat Missing Values dalam dataset:")
    print(null_count_df)
    print(f"Total missing values: {total_nulls}")

    # Tampilkan lokasi missing value: di kolom apa dan di baris ke berapa
    print("\nLokasi Missing Value per Kolom:")
    df_indexed = df.with_row_index("row_index")
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            lokasi = df_indexed.filter(pl.col(col).is_null()).select("row_index")
            print(f"\nKolom '{col}': {null_count} missing value, di baris index:")
            print(lokasi)
else:
    print("Tidak terdapat Missing Values dalam dataset.")

# HANDLING MISSING VALUE 

# 1. Menghapus missing values
print("\n" + "=" * 50)
print("HANDLING MISSING VALUE")
print("=" * 50)

# menghapus baris yang memiliki missing value pada kolom "Salary"
df_clean = df.drop_nulls(subset=["Salary"])
print(f"\nShape sebelum drop: {df.shape}")
print(f"Shape sesudah drop baris dengan Salary kosong: {df_clean.shape}")

# mengganti missing value pada kolom "Education Level" dan "Job Title" dengan modus (most frequent value)
modus_education = df_clean["Education Level"].mode()[0]
modus_job = df_clean["Job Title"].mode()[0]

# Isi kolom kategorikal dengan modusnya masing-masing
df_clean = df_clean.with_columns([
    pl.col("Education Level").fill_null(modus_education),
    pl.col("Job Title").fill_null(modus_job)
])
print(f"\nMengganti missing value pada kolom 'Education Level' dengan modus: {modus_education}")
print(f"Mengganti missing value pada kolom 'Job Title' dengan modus: {modus_job}")


# mengganti missing value pada kolom numerik "Age" dan "Years of Experience" dengan median
# karena median lebih robust terhadap outlier dibandingkan mean
median_age = df_clean["Age"].median()
median_experience = df_clean["Years of Experience"].median()

df_clean = df_clean.with_columns([
    pl.col("Age").fill_null(median_age),
    pl.col("Years of Experience").fill_null(median_experience)
])

print(f"\nMengganti missing value pada kolom 'Age' dengan median: {median_age}")
print(f"Mengganti missing value pada kolom 'Years of Experience' dengan median: {median_experience}")

# verifikasi: pastikan tidak ada missing value yang tersisa
print(f"\nSisa missing value setelah handling: {df_clean.null_count().sum_horizontal().item()}")
print(f"Shape dataset bersih: {df_clean.shape}")


# Cek outlier dengan metode IQR (Interquartile Range)
print("\n" + "=" * 50)
print("CEK OUTLIER DENGAN METODE IQR")
print("=" * 50)

# Deteksi Outlier Sebelum Handling
# outlier: nilai di luar rentang [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
total_outliers_before = {}
for col in numeric_cols:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Hitung jumlah baris outlier
    count = df_clean.filter(
        (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
    ).height
    total_outliers_before[col] = count
    if count > 0:
        print(
            f"Kolom '{col}': {count} outlier (Rentang Normal: {lower_bound:.2f} s/d"
            f" {upper_bound:.2f})"
        )

total_outliers_count = sum(total_outliers_before.values())
print(f"\nTotal outliers di semua kolom numerik: {total_outliers_count}")

# Handling outlier dengan IQR Capping
# nilai outlier di-cap (dipotong) ke batas rentang normal, sehingga baris data tidak dibuang
exprs_cap = []
for col in numeric_cols:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    exprs_cap.append(
        pl.col(col).clip(lower_bound=lower_bound, upper_bound=upper_bound)
    )

df_no_outliers = df_clean.with_columns(exprs_cap)

print(f"\nShape dataset setelah capping outlier: {df_no_outliers.shape}")

# Perbandingan Statistik Sebelum & Sesudah Capping
print("\nPerbandingan statistik sebelum & sesudah capping:")
for col in numeric_cols:
    if total_outliers_before[col] > 0:
        print(f"\nKolom: {col}")
        print(
            f"  Sebelum - Min: {df_clean[col].min():.2f}, Max: {df_clean[col].max():.2f}, Mean:"
            f" {df_clean[col].mean():.2f}"
        )
        print(
            f"  Sesudah - Min: {df_no_outliers[col].min():.2f}, Max:"
            f" {df_no_outliers[col].max():.2f}, Mean:"
            f" {df_no_outliers[col].mean():.2f}"
        )


# mulai menormalisasi data numerik
print("\n" + "=" * 50)
print("NORMALISASI DATA NUMERIK")
print("=" * 50)

# metode 1 Simple Feature Scaling
# melakukan scaling data menggunakan nilai maksimum sehingga nilai hasil transformasi berada dalam rentang yang lebih kecil.
# contoh 7,10,15,25; maka nilai yg maksimum diambil, kemudian semua nilai dibagi dengan 25, sehingga hasil transformasi menjadi 0.28, 0.4, 0.6, 1.0

print("\n SIMPLE FEATURE SCALING")
simple_scaling = df_no_outliers.with_columns([
    (pl.col(col) / pl.col(col).max()).alias(col)
    for col in numeric_cols
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
    for col in numeric_cols
])

print("\nHasil Min-Max Scaling:")
print(min_max_scaling.head())

# metode 3 Z-Score Normalization
# melakukan normalisasi data menggunakan rata-rata dan standar deviasi sehingga nilai hasil transformasi memiliki distribusi normal dengan rata-rata 0 dan standar deviasi 1.
print("\n Z-SCORE NORMALIZATION")
z_score_normalization = df_no_outliers.with_columns([
    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
    for col in numeric_cols
])
print("\nHasil Z-Score Normalization:")
print(z_score_normalization.head())


# contoh perbandingan hasil normalisasi pada kolom Salary
print("\nPerbandingan hasil normalisasi pada kolom 'Salary':")
print(f"\nSebelum Normalisasi (setelah capping outlier):\n{df_no_outliers.select('Salary').head(5)}")
print(f"\nSimple Feature Scaling:\n{simple_scaling.select('Salary').head(5)}")
print(f"\nMin-Max Scaling:\n{min_max_scaling.select('Salary').head(5)}")
print(f"\nZ-Score Normalization:\n{z_score_normalization.select('Salary').head(5)}")