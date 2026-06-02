# Optimasi Target Marketing Campaign Deposito Berjangka Menggunakan Machine Learning

## Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan model *Machine Learning* untuk mengoptimalkan strategi *marketing campaign* deposito berjangka di sebuah bank. Dengan memanfaatkan data historis, model ini didesain untuk memprediksi nasabah yang berpotensi melakukan deposit, serta mengidentifikasi faktor-faktor kunci yang mempengaruhi keputusan mereka.

## Latar Belakang Masalah
Industri perbankan menghadapi tantangan dalam memprediksi perilaku nasabah terkait produk deposito berjangka. Kesalahan prediksi dapat mengakibatkan pemborosan sumber daya dan hilangnya peluang bisnis yang berharga. Oleh karena itu, diperlukan pendekatan yang lebih cerdas dan berbasis data untuk menargetkan calon nasabah.

## Tujuan Proyek
1.  Mengembangkan model *Machine Learning* yang mampu memprediksi kemungkinan seorang nasabah akan melakukan deposito berjangka atau tidak.
2.  Mengidentifikasi faktor-faktor atau variabel yang paling berpengaruh terhadap keputusan nasabah untuk melakukan deposito, sehingga bank dapat merancang strategi pemasaran yang lebih efektif dan efisien.

## Metrik Evaluasi
Fokus utama proyek ini adalah **mereduksi *False Negatives*** (nasabah yang sebenarnya akan deposit tetapi tidak ditargetkan), karena ini merepresentasikan *opportunity cost* (kehilangan potensi pendapatan) yang tinggi bagi bank.

Metrik utama yang digunakan:
1.  **Recall (Sensitivity) untuk Kelas Positif (Deposit)**: Mengukur seberapa banyak nasabah yang benar-benar mau deposito berhasil diidentifikasi oleh model.
2.  **F2-Score**: Memberikan bobot dua kali lebih besar pada *Recall* dibandingkan *Precision*, sesuai dengan prioritas untuk meminimalkan *False Negatives*.
3.  **Confusion Matrix**: Untuk interpretasi visual dan bisnis mengenai kinerja model.

## Data
Dataset yang digunakan berisi informasi historis *marketing campaign* bank, mencakup 7805 entri unik dan 14 fitur setelah proses *cleaning* dan *feature engineering*.

**Fitur Kunci:**
| Attribute | Data Type | Description |
| --- | --- | --- |
| `age` | `int64` | Usia nasabah |
| `job` | `object` | Pekerjaan nasabah |
| `balance` | `int64` | Saldo rekening nasabah (USD) |
| `housing` | `object` | Kepemilikan kredit rumah |
| `loan` | `object` | Kepemilikan pinjaman dana |
| `contact` | `object` | Tipe media komunikasi yang digunakan menghubungi nasabah |
| `month` | `object` | Bulan terakhir kontak dalam satu tahun |
| `campaign` | `int64` | Jumlah kontak yang dilakukan selama kampanye ini untuk nasabah tersebut |
| `pdays` | `int64` | Jumlah hari sejak terakhir kali nasabah dihubungi pada kampanye sebelumnya |
| `poutcome` | `object` | Hasil dari kampanye pemasaran sebelumnya |
| `deposit` | `object` | Apakah nasabah sudah melakukan deposit atau belum (Target) |
| `pdays_contacted` | `int64` | Indikator apakah nasabah pernah dihubungi sebelumnya (0=belum, 1=sudah) |
| `pdays_bin` | `category` | Kategori rentang waktu sejak kontak terakhir |
| `age_group` | `category` | Kelompok usia nasabah |

## Metodologi

### 1. Data Cleaning & Feature Engineering
*   **Penanganan Duplikat**: 8 baris duplikat dihapus dari dataset awal.
*   **`pdays`**: Nilai `-1` (belum pernah dihubungi) diubah menjadi `np.nan` dan di-*engineer* menjadi fitur `pdays_contacted` (biner) dan `pdays_bin` (kategorikal rentang waktu).
*   **Kategorikal**: Kategori 'unknown' dan 'other' pada kolom `contact`, `poutcome`, dan `job` digabungkan atau dikategorikan ulang untuk simplifikasi dan kejelasan makna bisnis.
*   **`age_group`**: Fitur baru dibuat untuk mengelompokkan usia nasabah.
*   **Missing Value**: Hanya kolom `pdays` yang memiliki missing value yang ditangani dalam *preprocessing pipeline*.

### 2. Exploratory Data Analysis (EDA)
*   **Korelasi Spearman**: Digunakan untuk menganalisis hubungan antar fitur numerik, menunjukkan korelasi positif lemah antara `balance` dan `deposit`, serta korelasi negatif lemah antara `campaign`, `pdays` dan `deposit`.
*   **Distribusi Target**: Kolom `deposit` menunjukkan distribusi yang cukup seimbang (52% 'no', 48% 'yes'), mengurangi kebutuhan akan teknik *resampling* ekstrem.
*   **Uji Hipotesis (Chi-Square & Mann-Whitney U)**: Dilakukan untuk fitur-fitur kategorikal (`pdays_bin`, `poutcome`, `contact`, `job`, `month`, `age_group`) dan numerik (`balance`, `campaign`). Semua fitur ini menunjukkan hubungan yang signifikan secara statistik dengan keputusan deposit nasabah (`p < 0.05`).
    *   **`pdays_bin`**: Nasabah yang pernah dihubungi sebelumnya (`pdays_bin` selain `never_contacted`) memiliki probabilitas deposit yang jauh lebih tinggi.
    *   **`poutcome`**: Nasabah dengan riwayat kampanye 'success' memiliki probabilitas deposit sangat tinggi (~91.5%).
    *   **`contact`**: Kontak melalui `cellular` adalah yang paling efektif (~55% konversi).
    *   **`job`**: `Student`, `Retired`, dan `Unemployed` memiliki tingkat konversi deposit tertinggi.
    *   **`month`**: Bulan Maret, Desember, dan Oktober menunjukkan tingkat konversi tertinggi.
    *   **`age_group`**: `Older Adults`, `Seniors`, dan `Young Adults` menunjukkan probabilitas deposit tertinggi.
    *   **`balance` dan `campaign`**: Ada perbedaan signifikan dalam saldo dan jumlah kontak kampanye antara nasabah yang deposit dan yang tidak.

### 3. Machine Learning Modeling
*   **Pemilihan Model**: Beberapa model kandidat diantaranya: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, XGBoost, dan LightGBM.
*   **Benchmarking Awal**: Random Forest, XGBoost, dan LightGBM menunjukkan performa terbaik pada *F2-Score* di kondisi *default* (threshold 0.5).
*   **Hyperparameter Tuning**: Ketiga model teratas dioptimalkan menggunakan `RandomizedSearchCV` dengan `StratifiedKFold` dan `f2_scorer`.
*   **Optimasi Threshold**: *Threshold* prediksi dioptimalkan untuk setiap model secara *out-of-fold* untuk memaksimalkan *F2-Score*.

### 4. Pengujian Akhir dan Pemilihan Model Terbaik
*   Model **Random Forest** (dengan *threshold* 0.12) terpilih sebagai model terbaik dengan *Test F2-Score* tertinggi **0.8216**.
*   **Performa Model Terpilih (Random Forest Tuned - Threshold 0.12):**
    *   **F2-Score**: 0.8216
    *   **Recall**: 0.9973 (Hampir semua nasabah yang akan deposit berhasil diidentifikasi)
    *   **Precision**: 0.4819
    *   **False Negatives**: Hanya **2** (turun drastis dari 285 pada model default)

### 5. Analisis Cost-Benefit
*   **Model Default (Threshold 0.50)**:
    *   Total Manfaat Bersih: **$42,160**
    *   *True Positives*: 461 nasabah, menghasilkan $115,250
    *   *False Positives*: 184 nasabah, kerugian $1,840
    *   *False Negatives*: 285 nasabah, kehilangan potensi $71,250
*   **Model Optimized (Threshold 0.12)**:
    *   Total Manfaat Bersih: **$177,500**
    *   *True Positives*: 744 nasabah, menghasilkan $186,000
    *   *False Positives*: 800 nasabah, kerugian $8,000
    *   *False Negatives*: Hanya **2** nasabah, kehilangan potensi $500
*   **Peningkatan Manfaat Bersih**: Model Optimized menghasilkan **peningkatan sebesar $135,340** dibandingkan model default.

### 6. Feature Importance
Fitur paling berpengaruh pada prediksi model adalah:
1.  `num__balance` (saldo)
2.  `num__age` (usia)
3.  `onehot__poutcome_success` (hasil kampanye sebelumnya yang 'sukses')
4.  `num__campaign` (jumlah kontak)
5.  `onehot__contact_cellular` (metode kontak seluler)

## Kesimpulan
Proyek ini berhasil mengembangkan model **Random Forest** yang telah dioptimasi dengan *threshold* 0.12. Model ini sangat efektif dalam meminimalkan *False Negatives* (hanya 2 kasus), sesuai dengan tujuan bisnis untuk mengurangi *opportunity cost*. Peningkatan performa F2-Score dan Recall sangat signifikan, menghasilkan **peningkatan manfaat bersih sebesar $135,340** bagi bank. Fitur-fitur seperti saldo, usia, riwayat kampanye sukses, dan metode kontak seluler merupakan prediktor terpenting.

## Rekomendasi
1.  **Prioritaskan Segmen Potensial Tinggi**: Targetkan nasabah dengan riwayat `poutcome_success`, pekerjaan `Student`, `Retired`, `Unemployed`, kelompok usia `Older Adults`, `Seniors`, `Young Adults`, serta nasabah dengan saldo tinggi.
2.  **Optimalkan Saluran & Waktu Kontak**: Gunakan `cellular` sebagai saluran utama. Fokuskan kampanye pada bulan Maret, Desember, dan Oktober. Hindari kontak yang berlebihan (`campaign` tinggi).
3.  **Strategi untuk Nasabah Baru/Unknown**: Kembangkan strategi *awareness* atau penawaran khusus untuk nasabah yang belum pernah dihubungi atau memiliki riwayat 'unknown' untuk membangun kepercayaan.
4.  **Implementasi & Monitoring**: Integrasikan model yang dioptimasi ke sistem operasional pemasaran bank dan monitor performanya secara berkala.
5.  **Eksplorasi Lanjutan**: Pertimbangkan eksplorasi fitur baru atau interaksi fitur, serta tinjau ulang matriks *cost-benefit* secara berkala.
