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
modus_gender = df_clean["Gender"].mode()[0]

# Isi kolom kategorikal dengan modusnya masing-masing
df_clean = df_clean.with_columns([
    pl.col("Education Level").fill_null(modus_education),
    pl.col("Job Title").fill_null(modus_job),
    pl.col("Gender").fill_null(modus_gender).str.replace("Other", modus_gender)
])

print(f"\nMengganti missing value pada kolom 'Education Level' dengan modus: {modus_education}")
print(f"Mengganti missing value pada kolom 'Job Title' dengan modus: {modus_job}")
print(f"Mengganti missing value pada kolom 'Gender' dengan modus: {modus_gender}")

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


print("\nKeputusan: Outlier tidak akan dihapus dari dataset karena masih batas normal.")

print("\n" + "=" * 50)
print("STANDARDISASI & NORMALISASI")
print("=" * 50)

# cek nilai unik pada kolom "Education Level" sebelum dan sesudah normalisasi
print("Nilai unik sebelum:", df_clean["Education Level"].unique().sort().to_list())

df_clean = df_clean.with_columns(
    pl.col("Education Level")
    .str.replace("Bachelor's Degree", "Bachelor's")
    .str.replace("Master's Degree", "Master's")
    .str.replace("phD", "PhD")
)
print("Nilai unik sesudah:", df_clean["Education Level"].unique().sort().to_list())

# gabungkan job title yang jarang muncul (count < 30) menjadi "Other"
# agar asumsi chi-square (expected frequency >= 5) lebih terpenuhi
job_counts = df_clean["Job Title"].value_counts()
common_jobs = job_counts.filter(pl.col("count") >= 30)["Job Title"].to_list()

print(f"\nJumlah kategori Job Title sebelum penggabungan: {df_clean['Job Title'].n_unique()}")
df_clean = df_clean.with_columns(
    pl.when(pl.col("Job Title").is_in(common_jobs))
    .then(pl.col("Job Title"))
    .otherwise(pl.lit("Other"))
    .alias("Job Title")
)
print(f"Jumlah kategori Job Title sesudah penggabungan: {df_clean['Job Title'].n_unique()}")

# salary sebagai target, education level dan job title sebagai fitur kategorikal
from scipy.stats import chi2_contingency

s_low = df_clean["Salary"].quantile(1/3)
s_high = df_clean["Salary"].quantile(2/3)
df_clean = df_clean.with_columns(
    pl.when(pl.col("Salary") <= s_low).then(pl.lit("Low"))
    .when((pl.col("Salary") > s_low) & (pl.col("Salary") <= s_high)).then(pl.lit("Medium"))
    .otherwise(pl.lit("High"))
    .alias("Salary_Category")
)

print("\n" + "=" * 50)
print("SELEKSI FITUR KATEGORIKAL DENGAN CHI-SQUARE")
print("=" * 50)

# Seleksi Fitur kategorikal dengan Chi-Square
print(f"batas bawah salary: Low <= {s_low:.2f}, Medium > {s_low:.2f} and <= {s_high:.2f}, High > {s_high:.2f}")

categorical_cols = ["Gender", "Education Level", "Job Title"]
alpha = 0.05
chi_square_results = {}
for col in categorical_cols:
    # Buat tabel kontingensi
    contingency_df = (
        df_clean.group_by([col, "Salary_Category"])
        .len()
        .pivot(on="Salary_Category", index=col, values="len")
        .fill_null(0)
        .sort(col)
    )
    # kolom label dibuang hanya saat perhitungan chi-square (harus murni angka)
    contingency = contingency_df.drop(col).to_numpy()
    chi2, p, dof, expected = chi2_contingency(contingency)
    chi_square_results[col] = (chi2, p)

    print(f"\nTabel Kontingensi '{col}' vs Salary_Group:")
    print(contingency_df)
    print(f"Chi-Square Value: {chi2:.4f}")
    print(f"P-Value         : {p:.6f}")
    if p < alpha:
        print(f"-> p < {alpha}: '{col}' BERHUBUNGAN signifikan dengan Salary (fitur DIPERTAHANKAN)")
    else:
        print(f"-> p >= {alpha}: '{col}' TIDAK berhubungan signifikan dengan Salary (kandidat DIBUANG)")
        