# Praktikum Machine Learning

Kumpulan tugas praktikum mata kuliah Machine Learning — Semester 5, Fakultas Vokasi Universitas Airlangga.

Di repo ini aku nyatet perjalanan belajar ML dari bawah: mulai dari rapihin data berantakan, normalisasi, sampai nyeleksi fitur mana yang beneran ngaruh ke target. Semua diimplementasi pakai Python.

## Daftar Tugas

| Folder / File | Topik | Isi |
|---|---|---|
| `tm-2-preprocessing/` | Data Preprocessing | Cek & handling missing value, deteksi outlier IQR pada dataset salary & wine |
| `tm-3-normalisasi/` | Data Transformation | Normalisasi (Simple Feature Scaling, Min-Max, Z-Score) pada dataset wine, salary, dan Spotify |
| `tm-4-seleksi/` | Seleksi Fitur | Chi-Square untuk fitur kategorikal & ANOVA untuk fitur numerikal terhadap target Salary |
| `spotify-data-visualization.ipynb` | Data Visualization | Eksplorasi & visualisasi data lagu Spotify |

## Dataset

- `Salary_Data.csv` — gaji karyawan (Age, Gender, Education Level, Job Title, Years of Experience, Salary)
- `WineQT.csv` — kualitas wine dari 11 fitur kimiawi
- `songs_normalize.csv` — lagu-lagu hit Spotify 2000–2019

## Tools

- Python (polars untuk manipulasi data, scipy untuk uji statistik)
- Jupyter Notebook

## Penjelasan Tiap TM

TM (Tatap Muka) adalah pertemuan ke-N praktikum, dan tiap folder `tm-N` berisi pengerjaan tugas dari pertemuan tersebut.

### tm-2-preprocessing — Data Preprocessing
**Materinya:** sebelum data bisa diapapun, dia harus bersih dulu. Dua musuh utamanya: *missing value* (data kosong) dan *outlier* (nilai ekstrem).

**Yang dikerjakan:**
- `datasalary.py` — cek missing value per kolom, lalu ditangani sesuai karakternya: baris dengan target kosong di-drop, kolom kategorikal diisi modus, kolom numerik diisi median (lebih tahan banting terhadap outlier daripada mean)
- `datawine.py` — dataset wine yang sudah bersih dari missing value, lalu dideteksi outliernya pakai metode IQR dan ditangani dengan capping (nilai di luar batas dipotong ke batas terdekat, jadi nggak ada baris yang dibuang)

### tm-3-normalisasi — Data Transformation
**Materinya:** kolom-kolom numerik sering punya skala yang beda jauh (umur 20–60 vs gaji 35.000–250.000). Normalisasi menyamakan skalanya tanpa merusak informasi, biar adil buat algoritma ML di tahap selanjutnya.

**Yang dikerjakan:** tiga metode normalisasi diuji pada tiga dataset (`wine.py`, `salary.py`, `spotify.py`):
- *Simple Feature Scaling* — semua nilai dibagi nilai maksimum kolomnya
- *Min-Max* — dipetakan ke rentang [0, 1]
- *Z-Score* — diubah jadi distribusi dengan rata-rata 0 dan standar deviasi 1

### tm-4-seleksi — Seleksi Fitur
**Materinya:** nggak semua kolom di dataset beneran punya hubungan dengan target. Seleksi fitur bertanya: "fitur mana yang beneran ngaruh?" — dengan bukti statistik, bukan kira-kira.

**Yang dikerjakan:** `salary.py` menyeleksi 5 fitur terhadap target `Salary`:
- Fitur kategorikal (Gender, Education Level, Job Title) diuji pakai **Chi-Square** — target Salary numerik dikelompokkan dulu jadi Low/Medium/High sesuai aturan "numerik harus dikelompokkan kalau mau Chi-Square"
- Fitur numerikal (Age, Years of Experience) diuji pakai **ANOVA** — dibagi 4 kuartil, lalu diuji apakah rata-rata Salary beda signifikan antar kuartil (t-test nggak kepake karena cuma sanggup bandingin 2 kelompok)
- Hasilnya: kelima fitur signifikan (p < 0,05) dan layak dipertahankan

## Tools

- Python (polars untuk manipulasi data, scipy untuk uji statistik)
- Jupyter Notebook

## Catatan

- File laporan (.pdf/.docx) dan arsip pengumpulan tugas (.zip) sengaja tidak di-track di repo ini (lihat `.gitignore`)
- README ini ditulis dengan bantuan [Claude](https://claude.com/claude-code)
