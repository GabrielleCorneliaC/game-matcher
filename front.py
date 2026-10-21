import streamlit as st
import requests

st.set_page_config(page_title="Game Rating Predictor", page_icon="🎮")

st.title("✨ Game Rating Predictor")
st.markdown("Prediksi rating game berdasarkan data interaksi pemain menggunakan model KNN.")

# Input Form
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        plays = st.number_input("Total Plays", min_value=0, value=15000)
        backlogs = st.number_input("Backlogs", min_value=0, value=2000)
    with col2:
        playing = st.number_input("Currently Playing", min_value=0, value=500)
        wishlist = st.number_input("Wishlist", min_value=0, value=3000)
        
    genres = st.text_input("Genres (pisahkan dengan koma)", value="RPG, Adventure")
    
    submitted = st.form_submit_button("Prediksi Rating")

if submitted:
    # Memanggil endpoint lokal
    url = f"http://127.0.0.1:5001/api/predict"
    params = {
        'plays': plays,
        'playing': playing,
        'backlogs': backlogs,
        'wishlist': wishlist,
        'genres': genres
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('status') == 'success':
            st.success("Berhasil melakukan prediksi!")
            st.metric(label="Prediksi Rating KNN", value=data['predicted_rating'])
            
            st.subheader("5 Game dengan Karakteristik Serupa")
            for game in data['similar_games_reference']:
                with st.expander(f"{game['title']} (Rating: {game['score']})"):
                    st.write(f"**Genre:** {game['genres']}")
                    st.write(f"**Jarak Euclidean:** {game['distance']}")
        else:
            st.error(f"Error dari model: {data.get('message')}")
    except requests.exceptions.ConnectionError:
        st.error("Gagal terhubung ke API. Pastikan app.py (Flask) sedang berjalan!")