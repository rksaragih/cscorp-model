# 📊 Prediksi Biaya Produksi (Event Organizer - Production)

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk membangun model *machine learning* yang dapat memprediksi biaya produksi dalam konteks industri Event Organizer, khususnya pada kategori **production** (pembuatan booth, backdrop, signage, dan sejenisnya).

Model dikembangkan menggunakan pendekatan regresi untuk memperkirakan total biaya berdasarkan berbagai faktor seperti jenis produk, material, dimensi, kompleksitas, dan lainnya.

---

## 🎯 Tujuan

* Mengembangkan model prediksi biaya produksi
* Membandingkan performa beberapa algoritma (*Linear Regression* dan *Random Forest*)
* Mengimplementasikan model ke dalam aplikasi web berbasis Laravel
* Meningkatkan efisiensi estimasi biaya produksi

---

## 🧠 Metodologi

### 1. Dataset

Dataset yang digunakan merupakan **data dummy sintetis** yang dibuat menyerupai kondisi nyata pada industri produksi Event Organizer.

Fitur yang digunakan:

* `product_type`
* `material_type`
* `length_m`
* `width_m`
* `height_m`
* `volume_m3`
* `complexity_level`
* `quantity`
* `num_workers`
* `production_days`
* `finishing_type`
* `installation`
* `location_type`

Target:

* `total_cost`

---

### 2. Preprocessing

* Encoding variabel kategorikal menggunakan **OneHotEncoder**
* Penggabungan preprocessing dan model menggunakan **Pipeline** untuk menjaga konsistensi

---

### 3. Model yang Digunakan

* **Linear Regression** (baseline model)
* **Random Forest Regression** (model utama)

---

### 4. Evaluasi Model

Metrik yang digunakan:

* MAE (*Mean Absolute Error*)
* RMSE (*Root Mean Squared Error*)
* R² Score
* MAPE (*Mean Absolute Percentage Error*)

---

## 📊 Hasil Evaluasi

| Model             | MAE      | RMSE     | R²   |
| ----------------- | -------- | -------- | ---- |
| Linear Regression | 3.01e+08 | 4.24e+08 | 0.77 |
| Random Forest     | 1.30e+08 | 2.38e+08 | 0.93 |

**MAPE (Random Forest): 21.78%**

### 🔍 Insight

* Random Forest memberikan performa yang jauh lebih baik dibanding Linear Regression
* Data memiliki pola **non-linear**
* Model mampu menjelaskan hingga **92% variasi data**
* Tingkat kesalahan rata-rata sekitar **21%**

---

## ⚙️ Arsitektur Sistem

```text
Frontend (Laravel Blade)
        ↓
Laravel Backend (Controller)
        ↓
Python API (Flask)
        ↓
Machine Learning Model (.pkl)
```

---

## 🚀 Cara Menjalankan Project

### 1. Setup Python API

```bash
cd ml-api
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Jalankan API:

```bash
cd src
python app.py
```

---

### 2. Akses Aplikasi

* API: http://127.0.0.1:5000

---

## 🧩 Fitur Utama

* Input spesifikasi produksi melalui form
* Prediksi biaya produksi secara otomatis
* Estimasi jumlah pekerja dan durasi produksi (rule-based)
* Integrasi Laravel dengan Python API

---

## 🧠 Insight Tambahan

* Model di-deploy sebagai REST API (microservice)
* Tidak semua input berasal dari user (beberapa dihitung otomatis)
* Kombinasi **Machine Learning + Rule-Based System**

---

## 📚 Teknologi yang Digunakan

* Python (pandas, scikit-learn, Flask)
* Joblib (model serialization)

---

## 📄 Catatan

Proyek ini menggunakan dataset sintetis (dummy) karena keterbatasan akses terhadap data asli dari mitra.

---