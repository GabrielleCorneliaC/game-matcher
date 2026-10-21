import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import ast

df = pd.read_csv('games.csv')
# ... (kode pembersihan data & encoding genre yang SAMA PERSIS dengan di app.py) ...
# 1. MEMBACA DAN MEMBERSIHKAN DATASET 
print("Membaca dataset...")
df = pd.read_csv('games.csv')

df = df.rename(columns={
    'Unnamed: 0': 'id', 
    'Title': 'title', 
    'Rating': 'score'
})

df = df.dropna(subset=['id', 'title', 'Genres', 'score', 'Plays', 'Playing', 'Backlogs', 'Wishlist'])
df = df.drop_duplicates(subset=['title']) 
df = df.reset_index(drop=True)

# 2. FUNGSI KONVERSI HURUF 'K' (RIBUAN) MENJADI ANGKA
def bersihkan_angka_k(nilai):
    if isinstance(nilai, str):
        if 'K' in nilai:
            return float(nilai.replace('K', '')) * 1000
        try:
            return float(nilai)
        except:
            return 0.0
    return float(nilai) if pd.notnull(nilai) else 0.0

df['Plays'] = df['Plays'].apply(bersihkan_angka_k)
df['Playing'] = df['Playing'].apply(bersihkan_angka_k)
df['Backlogs'] = df['Backlogs'].apply(bersihkan_angka_k)
df['Wishlist'] = df['Wishlist'].apply(bersihkan_angka_k)

# 3. PARSING TEKS GENRE (STRING) MENJADI LIST 
def ubah_teks_genre_ke_list(teks_mentah):
    if not isinstance(teks_mentah, str):
        return []
    
    teks_bersih = teks_mentah.replace("[", "").replace("]", "")
    teks_bersih = teks_bersih.replace("'", "").replace('"', "")
    
    daftar_genre = []
    for g in teks_bersih.split(","):
        nama_genre = g.strip()  
        if nama_genre:          
            daftar_genre.append(nama_genre)
            
    return daftar_genre

df['Genres_List'] = df['Genres'].apply(ubah_teks_genre_ke_list)

# 4. PROSES ENCODE GENRE MANUAL
semua_genre = []
for daftar_genre in df['Genres_List']:
    for g in daftar_genre:
        if g not in semua_genre:
            semua_genre.append(g)
semua_genre = sorted(semua_genre)

# bikin matriks biner (0 dan 1) 
matrix_genre = []
for daftar_genre in df['Genres_List']:
    baris_biner = []
    for g in semua_genre:
        if g in daftar_genre:
            baris_biner.append(1)  # 1 jika game memiliki genre tersebut
        else:
            baris_biner.append(0)  # 0 jika tidak
    matrix_genre.append(baris_biner)

genre_df = pd.DataFrame(matrix_genre, columns=semua_genre)
df_angka = df[['Plays', 'Playing', 'Backlogs', 'Wishlist']].copy()
df_angka_scaled = (df_angka - df_angka.min()) / (df_angka.max() - df_angka.min())

# Gabungkan data untuk proses evaluasi
X = pd.concat([df_angka_scaled, genre_df], axis=1)
y = df['score']

# -----------

# 2. EVALUASI METRIK NORMAL (MAE & RMSE)
knn_uni = KNeighborsRegressor(n_neighbors=5, metric='euclidean', weights='uniform')
knn_uni.fit(X, y)
pred_uni = knn_uni.predict(X)

knn_dist = KNeighborsRegressor(n_neighbors=5, metric='euclidean', weights='distance')
knn_dist.fit(X, y)
pred_dist = knn_dist.predict(X) # Hanya untuk CV

mae = mean_absolute_error(y, pred_uni)
rmse = np.sqrt(mean_squared_error(y, pred_uni))

# 3. CROSS-VALIDATION (5-FOLD)
cv_scores_uni = -cross_val_score(knn_uni, X, y, cv=5, scoring='neg_mean_squared_error')
cv_scores_dist = -cross_val_score(knn_dist, X, y, cv=5, scoring='neg_mean_squared_error')

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_logs = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    fold_logs.append({
        'Fold': f'Fold {i+1}',
        'Jumlah_Data_Testing': len(test_index),
        'Indeks_Data_Testing_Awal': test_index[0],
        'Indeks_Data_Testing_Akhir': test_index[-1]
    })

# 4. SIMPAN 
with pd.ExcelWriter('Evaluasi_Model_plz.xlsx', engine='openpyxl') as writer:
    # Summary Sheet
    pd.DataFrame({
        'Metrik': ['MAE', 'RMSE', 'CV RMSE (Uniform)', 'CV RMSE (Distance)'],
        'Nilai': [mae, rmse, np.sqrt(cv_scores_uni).mean(), np.sqrt(cv_scores_dist).mean()],
        'Formula_Excel': [
            '=AVERAGE(E2:E1000)', 
            '=SQRT(AVERAGE(F2:F1000))', 
            'Rata-rata 5-fold (Uniform)', 
            'Rata-rata 5-fold (Weighted)'
        ]
    }).to_excel(writer, sheet_name='Summary_Evaluasi', index=False)
    
    # Detail CV Fold
    pd.DataFrame({
        'Fold': [1, 2, 3, 4, 5],
        'Uniform_RMSE': np.sqrt(cv_scores_uni),
        'Distance_RMSE': np.sqrt(cv_scores_dist)
    }).to_excel(writer, sheet_name='Detail_CV_Fold', index=False)
    
    # Langkah Perhitungan Manual
    df_detail = df[['title', 'score']].copy()
    df_detail['Prediksi'] = pred_uni
    df_detail['Selisih (y - y_pred)'] = df_detail['score'] - df_detail['Prediksi']
    df_detail['Selisih_Absolut'] = np.abs(df_detail['Selisih (y - y_pred)'])
    df_detail['Selisih_Kuadrat'] = df_detail['Selisih (y - y_pred)']**2
    df_detail.to_excel(writer, sheet_name='Langkah_Perhitungan', index=False)

    pd.DataFrame(fold_logs).to_excel(writer, sheet_name='Logika_Pembagian_CV', index=False)

print("Berhasil! File Evaluasi_Model_KNN.xlsx sudah dibuat.")