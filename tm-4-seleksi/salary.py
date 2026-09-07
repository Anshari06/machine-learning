import polars as pl
from polars import selectors as cs

df = pl.read_csv(r"C:\\Users\\LENOVO\\OneDrive\\Documents\\TUGAS\\SEM V\\Machine Learnig\\Salary_Data.csv", schema_overrides={"Years of Experience": pl.Float64})

print(df)
print(f"Shape dataset: {df.shape}")
print(df.describe())

# cek missing value
print("\n" + "=" * 50)
print("CEK & HANDLE MISSING VALUE")
print("=" * 50)

null_count_df = df.null_count()
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

# drop missing values salary
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

if df_clean.null_count().sum_horizontal().item() == 0:
    print(f"\nSemua missing value telah berhasil ditangani. Total remaining nulls: {df_clean.null_count().sum_horizontal().item()}")

# cek outlier pada kolom numerik menggunakan IQR
print("\n" + "=" * 50)
print("CEK OUTLIER")
print("=" * 50)

print("\nMendeteksi outlier pada kolom numerik menggunakan metode IQR (Interquartile Range).")

numeric_cols = df.select(cs.numeric()).columns
total_outliers_before = {}
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    count = df_clean.filter(
            (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
    ).height

    print("\n" + "=" * 50)
    print(f"\nKolom: {col}")
    print(f"Q1           : {Q1}")
    print(f"Q3           : {Q3}")
    print(f"IQR          : {IQR}")
    print(f"Lower Bound  : {lower_bound}")
    print(f"Upper Bound  : {upper_bound}")
    print(f"Jumlah Outlier: {count}")

    total_outliers_before[col] = count
    if count > 0:
        print(
            f"Kolom '{col}': {count} outlier (Rentang Normal: {lower_bound:.2f} s/d"
            f" {upper_bound:.2f})"
        )

total_outliers_count = sum(total_outliers_before.values())
print(f"\nTotal outliers di semua kolom numerik: {total_outliers_count}")
