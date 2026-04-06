import streamlit as st
import numpy as np
import pickle
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🎨 Page config
st.set_page_config(page_title="Mood Melody 🎵")

st.title("🎵 Mood Melody")
st.write("Get music recommendations based on your emotions 💡")

# 📁 Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "emotion_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

# ⚙️ Model config (must match training)
MAX_LEN = 35
DICT_SIZE = 15000
EMBEDDING_DIM = 100

# 🎯 Labels
emotion_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# 🚀 Load model
@st.cache_resource
def load_model():
    model = Sequential()

    model.add(Input(shape=(MAX_LEN,)))

    model.add(
        Embedding(
            input_dim=DICT_SIZE,
            output_dim=EMBEDDING_DIM,
            trainable=False
        )
    )

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(6, activation='softmax'))

    model.load_weights(WEIGHTS_PATH)

    return model

# 🚀 Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

# 🔮 Predict function
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(padded)
    return emotion_labels[np.argmax(pred)]

# 🔐 Spotify credentials (PUT YOURS HERE)
CLIENT_ID = "2fc21d89c44e49d5a926a48807bf0545"
CLIENT_SECRET = "4616a1ab31ad470fbdcecc7aba8e7ef1"

@st.cache_resource
def get_spotify():
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    ))

# 🎵 Emotion → music mapping
emotion_music_map = {
    "joy": "happy upbeat songs",
    "sadness": "sad songs",
    "anger": "rock intense",
    "fear": "calm soothing",
    "love": "romantic songs",
    "surprise": "party songs"
}

# 🎶 Get songs
def get_songs(emotion):
    try:
        query = emotion_music_map.get(emotion, "trending songs")
        results = sp.search(q=query, type='track', limit=5)

        songs = []
        for track in results['tracks']['items']:
            songs.append({
                "name": track['name'],
                "artist": track['artists'][0]['name'],
                "url": track['external_urls']['spotify'],
                "image": track['album']['images'][0]['url']
            })

        return songs

    except Exception as e:
        st.error(f"Spotify error: {e}")
        return []

# 🔄 Load everything
model = load_model()
tokenizer = load_tokenizer()
sp = get_spotify()

# 📝 User input
user_input = st.text_area("Enter how you feel:")

# 🎯 Button action
if st.button("Recommend Music 🎶"):

    if not user_input.strip():
        st.warning("Please enter something")
    else:
        with st.spinner("Analyzing your mood..."):
            emotion = predict_emotion(user_input)

        st.success(f"Detected Emotion: {emotion}")

        songs = get_songs(emotion)

        if songs:
            for song in songs:
                st.image(song["image"], width=200)
                st.write(f"🎵 {song['name']} - {song['artist']}")
                st.markdown(f"[Listen on Spotify]({song['url']})")
                st.write("---")
        else:
            st.warning("No songs found")