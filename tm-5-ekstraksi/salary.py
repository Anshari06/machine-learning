import polars as pl

df = pl.read_csv("salary_preprocessed.csv")

print(f"Shape dataset: {df.shape}")
print(df.describe())
print(df)

# hapus kolom yang tidak diperlukan
df = df.drop(["Salary_Category", "Job Title"])
print(f"\ndataset setelah menghapus kolom yang tidak diperlukan: {df.shape}")

# hot encode kolom kategorikal
df_encoded = df.to_dummies(columns=["Education Level", "Gender"])
print(f"\ndataset setelah hot encoding: {df_encoded.shape}")

# pisahkan fitur dan target
X = df_encoded.drop("Salary") #fitur
y = df_encoded["Salary"] #target
X = X.to_numpy() #fitur dalam bentuk numpy array

