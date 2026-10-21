# 80, 200

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

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
        nilai_upper = nilai.upper()
        if 'K' in nilai_upper:
            return float(nilai_upper.replace('K', '')) * 1000
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
            baris_biner.append(1)  
        else:
            baris_biner.append(0)  
    matrix_genre.append(baris_biner)

genre_df = pd.DataFrame(matrix_genre, columns=semua_genre)

# 5. NORMALISASI FITUR ANGKA SECARA MANUAL (RUMUS MIN-MAX SCALING)
df_angka = df[['Plays', 'Playing', 'Backlogs', 'Wishlist']].copy()

min_asli = df_angka.min()
max_asli = df_angka.max()
print("\n=== BATAS MIN & MAX DATASET ===")
print("NILAI MINIMAL:")
print(min_asli)
print("\nNILAI MAKSIMAL:")
print(max_asli)
print("===============================\n")

df_angka_scaled = (df_angka - min_asli) / (max_asli - min_asli)
X = pd.concat([df_angka_scaled, genre_df], axis=1)
y = df['score']


# --- TAHAP 1: EVALUASI AKADEMIS (MENGGUNAKAN SPLIT 80/20) ---
print("=== MENJALANKAN EVALUASI (80/20) & EXCEL REPORT ===")

# Membagi Data Khusus untuk Evaluasi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih Model Evaluasi Khusus (Hanya 80%)
knn_eval_uni = KNeighborsRegressor(n_neighbors=5, metric='euclidean', weights='uniform')
knn_eval_uni.fit(X_train, y_train)
pred_uni = knn_eval_uni.predict(X_test) # Uji murni pada 20% data

knn_eval_dist = KNeighborsRegressor(n_neighbors=5, metric='euclidean', weights='distance')
knn_eval_dist.fit(X_train, y_train)

# Hitung Skor Valid MAE & RMSE Tunggal
mae = mean_absolute_error(y_test, pred_uni)
rmse = np.sqrt(mean_squared_error(y_test, pred_uni))

# Uji Kestabilan Lewat 5-Fold Cross Validation
cv_scores_uni = -cross_val_score(knn_eval_uni, X, y, cv=5, scoring='neg_mean_squared_error')
cv_scores_dist = -cross_val_score(knn_eval_dist, X, y, cv=5, scoring='neg_mean_squared_error')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_logs = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    fold_logs.append({
        'Fold': f'Fold {i+1}',
        'Jumlah_Data_Testing': len(test_index),
        'Indeks_Data_Testing_Awal': test_index[0],
        'Indeks_Data_Testing_Akhir': test_index[-1]
    })

# Export ke Excel
with pd.ExcelWriter('Evaluasi_Model_plz.xlsx', engine='openpyxl') as writer:
    pd.DataFrame({
        'Metrik': ['MAE (80/20)', 'RMSE (80/20)', 'CV RMSE (Uniform)', 'CV RMSE (Distance)'],
        'Nilai': [mae, rmse, np.sqrt(cv_scores_uni).mean(), np.sqrt(cv_scores_dist).mean()],
        'Formula_Excel': [
            '=AVERAGE(E2:E1000)', 
            '=SQRT(AVERAGE(F2:F1000))', 
            'Rata-rata 5-fold (Uniform)', 
            'Rata-rata 5-fold (Weighted)'
        ]
    }).to_excel(writer, sheet_name='Summary_Evaluasi', index=False)
    
    pd.DataFrame({
        'Fold': [1, 2, 3, 4, 5],
        'Uniform_RMSE': np.sqrt(cv_scores_uni),
        'Distance_RMSE': np.sqrt(cv_scores_dist)
    }).to_excel(writer, sheet_name='Detail_CV_Fold', index=False)
    
    # Menampilkan 20% sampel pengujian di Excel
    df_detail = df.loc[X_test.index, ['title', 'score']].copy()
    df_detail['Prediksi'] = pred_uni
    df_detail['Selisih (y - y_pred)'] = df_detail['score'] - df_detail['Prediksi']
    df_detail['Selisih_Absolut'] = np.abs(df_detail['Selisih (y - y_pred)'])
    df_detail['Selisih_Kuadrat'] = df_detail['Selisih (y - y_pred)']**2
    df_detail.to_excel(writer, sheet_name='Langkah_Perhitungan', index=False)

    pd.DataFrame(fold_logs).to_excel(writer, sheet_name='Logika_Pembagian_CV', index=False)

print("Berhasil! File Evaluasi Excel (80/20) sudah dibuat.")
print(f"Hasil Tes internal -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")
print("===============================================================\n")


# --- TAHAP LIVE API WEBSITE LARAVEL ---
# Latih ulang model baru 
knn_utama = KNeighborsRegressor(n_neighbors=5, metric='euclidean', weights='uniform')
knn_utama.fit(X.values, y.values)

@app.route('/api/predict', methods=['GET'])
def predict_rating():
    try:
        input_plays = bersihkan_angka_k(request.args.get('plays', '0'))
        input_playing = bersihkan_angka_k(request.args.get('playing', '0'))
        input_backlogs = bersihkan_angka_k(request.args.get('backlogs', '0'))
        input_wishlist = bersihkan_angka_k(request.args.get('wishlist', '0'))
        
        input_genres_str = request.args.get('genres', '')
        user_genres = [g.strip() for g in input_genres_str.split(',')] if input_genres_str else []

        inputs_dict = {
            'Plays': input_plays, 'Playing': input_playing, 
            'Backlogs': input_backlogs, 'Wishlist': input_wishlist
        }
        
        for kolom, nilai in inputs_dict.items():
            if nilai < min_asli[kolom] or nilai > max_asli[kolom]:
                return jsonify({
                    'status': 'error', 
                    'message': f"Nilai {kolom} tidak valid! Harus di antara {min_asli[kolom]} sampai {max_asli[kolom]}."
                }), 400

        scaled_plays = (input_plays - min_asli['Plays']) / (max_asli['Plays'] - min_asli['Plays'])
        scaled_playing = (input_playing - min_asli['Playing']) / (max_asli['Playing'] - min_asli['Playing'])
        scaled_backlogs = (input_backlogs - min_asli['Backlogs']) / (max_asli['Backlogs'] - min_asli['Backlogs'])
        scaled_wishlist = (input_wishlist - min_asli['Wishlist']) / (max_asli['Wishlist'] - min_asli['Wishlist'])
        
        vector_angka = [scaled_plays, scaled_playing, scaled_backlogs, scaled_wishlist]

        vector_genre = []
        for g in semua_genre:
            if g in user_genres:
                vector_genre.append(1)
            else:
                vector_genre.append(0)

        vektor_uji_lengkap = np.array(vector_angka + vector_genre).reshape(1, -1)

        prediksi_rating = knn_utama.predict(vektor_uji_lengkap)[0]

        # Cari 5 game terdekat dari 100% populasi data
        distances, indices = knn_utama.kneighbors(vektor_uji_lengkap, n_neighbors=5)
        
        daftar_game_tetangga = []
        for i in range(5):
            idx = indices[0][i] 
            
            daftar_game_tetangga.append({
                'title': str(df.iloc[idx]['title']),
                'score': float(df.iloc[idx]['score']),
                'genres': ", ".join(df.iloc[idx]['Genres_List']),
                'distance': round(float(distances[0][i]), 4)
            })

        return jsonify({
            'status': 'success',
            'predicted_rating': round(float(prediksi_rating), 2),
            'similar_games_reference': daftar_game_tetangga
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)