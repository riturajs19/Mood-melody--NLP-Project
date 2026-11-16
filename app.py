import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🎨 Page config
st.set_page_config(page_title="Mood Melody 🎵", page_icon="🎵", layout="centered")

st.title("🎵 Mood Melody")
st.write("Emotion-Aware Music Recommendation using Stacked BiLSTM, GloVe & Spotify Audio Profiling 💡")

# 📁 Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "emotion_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

# ⚙️ Model config (must match training)
MAX_LEN = 35
DICT_SIZE = 15000
EMBEDDING_DIM = 100

# 🎯 FIXED Label Mapping (Matches MoodMelodyNLP_Project.ipynb training order!)
# Original notebook: {'anger': 0, 'love': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5}
emotion_labels = {
    0: "anger",
    1: "love",
    2: "fear",
    3: "joy",
    4: "sadness",
    5: "surprise"
}
label_to_idx = {v: k for k, v in emotion_labels.items()}

# 🛡️ UPGRADE 1: Contraction Expansion & Negation-Aware Stopword Removal
# Preserves polarity shifters ('not', 'never', 'no', etc.) instead of deleting them!
NEGATION_WHITELIST = {
    'not', 'no', 'nor', 'never', 'neither', 'barely', 'hardly',
    'scarcely', 'without', 'against', 'cannot', 'too', 'very'
}

BASE_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
    'own', 'same', 'so', 'than', 's', 't', 'can', 'will', 'just', 'should', 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y'
}
STOPWORDS = BASE_STOPWORDS - NEGATION_WHITELIST

CONTRACTIONS = {
    r"won't": "will not", r"can't": "cannot", r"n't": " not",
    r"'re": " are", r"'s": " is", r"'d": " would", r"'ll": " will",
    r"'t": " not", r"'ve": " have", r"'m": " am",
    r"\bdidnt\b": "did not", r"\bdont\b": "do not", r"\bcant\b": "cannot",
    r"\bwont\b": "will not", r"\bisnt\b": "is not", r"\bwasnt\b": "was not",
    r"\bwerent\b": "were not", r"\bhavent\b": "have not", r"\bhasnt\b": "has not",
    r"\bhadnt\b": "had not", r"\bwouldnt\b": "would not", r"\bshouldnt\b": "should not",
    r"\bcouldnt\b": "could not", r"\bim\b": "i am", r"\bive\b": "i have",
    r"\bid\b": "i would", r"\byouve\b": "you have", r"\btheyve\b": "they have"
}


def text_preprocess(text, stop_words=True):
    """
    Expands contractions, strips punctuation, and removes stopwords
    while preserving NEGATION_WHITELIST ('not', 'never', 'no').
    """
    text = str(text).lower()
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    if stop_words:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# 🎛️ UPGRADE 2: Continuous Audio Feature Profiles (Russell's 2D Valence-Energy Circumplex Model)
AUDIO_PROFILES = {
    "joy":      {"valence": 0.85, "energy": 0.80, "tempo": 122, "emoji": "😄"},
    "sadness":  {"valence": 0.20, "energy": 0.25, "tempo": 78,  "emoji": "😢"},
    "anger":    {"valence": 0.25, "energy": 0.92, "tempo": 138, "emoji": "😡"},
    "love":     {"valence": 0.72, "energy": 0.48, "tempo": 96,  "emoji": "❤️"},
    "fear":     {"valence": 0.55, "energy": 0.18, "tempo": 68,  "emoji": "😨"},
    "surprise": {"valence": 0.78, "energy": 0.85, "tempo": 126, "emoji": "😲"}
}

# 🚀 Load model
@st.cache_resource
def load_model():
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=DICT_SIZE, output_dim=EMBEDDING_DIM, trainable=False),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(128)),
        Dense(6, activation='softmax')
    ])
    model.load_weights(WEIGHTS_PATH)
    return model


# 🚀 Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        loaded = pickle.load(f)
        if isinstance(loaded, dict) and "tokenizer" in loaded:
            return loaded["tokenizer"]
        return loaded


# 🔮 UPGRADE 3: Negation-Aware Prediction + Top-2 Softmax Blending
def predict_emotion(text):
    tokens = text_preprocess(text, stop_words=True)
    processed_str = " ".join(tokens)

    seq = tokenizer.texts_to_sequences([processed_str])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='pre', truncating='post')

    probs = model.predict(padded, verbose=0)[0].astype(float)

    # Negation-aware polarity calibration (handles negations like 'not happy' even if weights predate retraining)
    negation_set = {'not', 'no', 'never', 'neither', 'barely', 'hardly', 'without', 'cannot'}
    pos_cues = {'happy', 'joy', 'glad', 'excited', 'good', 'great', 'wonderful', 'love', 'sweet', 'amazing'}
    neg_cues = {'sad', 'humiliated', 'angry', 'scared', 'afraid', 'lonely', 'hopeless', 'bad', 'hurt'}

    for i, tok in enumerate(tokens):
        if i > 0 and any(tokens[j] in negation_set for j in range(max(0, i - 3), i)):
            if tok in pos_cues:
                # Flip false positive joy/love to sadness
                shift = probs[label_to_idx["joy"]] * 0.80 + probs[label_to_idx["love"]] * 0.60
                probs[label_to_idx["joy"]] *= 0.20
                probs[label_to_idx["love"]] *= 0.40
                probs[label_to_idx["sadness"]] += shift
            elif tok in neg_cues:
                # Flip negated sadness/fear ('did not feel humiliated/sad') toward joy/relief
                shift = probs[label_to_idx["sadness"]] * 0.65 + probs[label_to_idx["fear"]] * 0.50
                probs[label_to_idx["sadness"]] *= 0.35
                probs[label_to_idx["fear"]] *= 0.50
                probs[label_to_idx["joy"]] += shift

    # Re-normalize probability distribution
    probs = probs / np.sum(probs)

    # Top-2 Softmax Probability Blending (instead of blind np.argmax)
    top2_indices = np.argsort(probs)[::-1][:2]
    primary_idx, secondary_idx = int(top2_indices[0]), int(top2_indices[1])
    primary_emotion = emotion_labels[primary_idx]
    secondary_emotion = emotion_labels[secondary_idx]
    p1, p2 = float(probs[primary_idx]), float(probs[secondary_idx])

    # Detect mixed/ambiguous emotion if runner-up is >= 22% or within 15% of top emotion
    is_mixed = (p2 >= 0.22) or ((p1 - p2) < 0.15)

    # Compute probability-weighted Spotify audio feature targets
    target_valence = sum(probs[i] * AUDIO_PROFILES[emotion_labels[i]]["valence"] for i in range(6))
    target_energy = sum(probs[i] * AUDIO_PROFILES[emotion_labels[i]]["energy"] for i in range(6))
    target_tempo = sum(probs[i] * AUDIO_PROFILES[emotion_labels[i]]["tempo"] for i in range(6))

    prob_dict = {emotion_labels[i]: float(probs[i]) for i in range(6)}

    return {
        "primary_emotion": primary_emotion,
        "primary_conf": p1,
        "secondary_emotion": secondary_emotion if is_mixed else None,
        "secondary_conf": p2 if is_mixed else 0.0,
        "is_mixed": is_mixed,
        "probabilities": prob_dict,
        "tokens": tokens,
        "audio_targets": {
            "valence": round(target_valence, 2),
            "energy": round(target_energy, 2),
            "tempo": int(round(target_tempo))
        }
    }


# 🔐 Spotify credentials (reads from env or falls back to configured project credentials)
CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "2fc21d89c44e49d5a926a48807bf0545")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "4616a1ab31ad470fbdcecc7aba8e7ef1")

@st.cache_resource
def get_spotify():
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    ))


# 🎵 Emotion → music mapping (Match Mode vs. Uplift/Soothe Mode)
emotion_music_map = {
    "joy": "happy upbeat feel good songs",
    "sadness": "sad acoustic melancholic songs",
    "anger": "rock intense cathartic workout",
    "fear": "calm soothing ambient piano",
    "love": "romantic acoustic love songs",
    "surprise": "upbeat electronic party songs"
}

uplift_music_map = {
    "joy": "happy upbeat feel good songs",
    "sadness": "uplifting hopeful acoustic indie",
    "anger": "calm grounding peaceful instrumental",
    "fear": "calm soothing ambient meditation",
    "love": "warm romantic acoustic songs",
    "surprise": "feel good groove pop"
}


# 🎶 Get songs (supports Top-2 Mixed Mood Blending & Psychological Regulation Mode)
def get_songs(primary_emotion, secondary_emotion=None, mode="Match My Mood (Iso-Principle)"):
    try:
        active_map = emotion_music_map if "Match" in mode else uplift_music_map
        primary_query = active_map.get(primary_emotion, "trending songs")

        songs = []
        if secondary_emotion and "Match" in mode:
            # Blend 3 tracks from primary emotion + 2 tracks from secondary emotion
            res1 = sp.search(q=primary_query, type='track', limit=3)['tracks']['items']
            res2 = sp.search(q=active_map.get(secondary_emotion, "chill songs"), type='track', limit=2)['tracks']['items']
            raw_tracks = res1 + res2
        else:
            raw_tracks = sp.search(q=primary_query, type='track', limit=5)['tracks']['items']

        for track in raw_tracks:
            songs.append({
                "id": track['id'],
                "name": track['name'],
                "artist": track['artists'][0]['name'],
                "url": track['external_urls']['spotify'],
                "image": track['album']['images'][0]['url'] if track['album']['images'] else ""
            })

        return songs

    except Exception as e:
        st.error(f"Spotify error: {e}")
        return []


# 🔄 Load everything
model = load_model()
tokenizer = load_tokenizer()
sp = get_spotify()

# 📝 User input & Therapy Mode Toggle
user_input = st.text_area(
    "Enter how you feel:",
    placeholder="e.g., I am not happy at all today, everything is falling apart..."
)

mode = st.radio(
    "🎧 Recommendation Strategy (Music Psychology):",
    ["Match My Mood (Iso-Principle)", "Uplift / Soothe My Mood (Mood Regulation)"],
    horizontal=True
)

# 🎯 Button action
if st.button("Recommend Music 🎶", type="primary"):
    if not user_input.strip():
        st.warning("Please enter something")
    else:
        with st.spinner("Analyzing your mood with Stacked BiLSTM..."):
            result = predict_emotion(user_input)

        emo = result["primary_emotion"]
        sec_emo = result["secondary_emotion"]
        emoji = AUDIO_PROFILES[emo]["emoji"]

        if result["is_mixed"] and sec_emo:
            sec_emoji = AUDIO_PROFILES[sec_emo]["emoji"]
            st.success(
                f"**Detected Blended Mood:** {emoji} **{emo.capitalize()}** ({result['primary_conf']:.1%}) "
                f"+ {sec_emoji} **{sec_emo.capitalize()}** ({result['secondary_conf']:.1%})"
            )
        else:
            st.success(f"**Detected Emotion:** {emoji} **{emo.capitalize()}** ({result['primary_conf']:.1%} confidence)")

        # Display Preprocessed Tokens & Continuous Audio Feature Targets
        st.caption(f"🔍 **Preprocessed BiLSTM Tokens (negations preserved):** `{result['tokens']}`")

        col1, col2, col3 = st.columns(3)
        col1.metric("Target Valence (Positivity)", f"{result['audio_targets']['valence']}")
        col2.metric("Target Energy (Intensity)", f"{result['audio_targets']['energy']}")
        col3.metric("Target Tempo", f"{result['audio_targets']['tempo']} BPM")

        with st.expander("📊 View Full Softmax Emotion Probability Distribution"):
            prob_df = pd.DataFrame.from_dict(
                result["probabilities"], orient="index", columns=["Probability"]
            )
            st.bar_chart(prob_df)

        songs = get_songs(emo, secondary_emotion=sec_emo, mode=mode)

        if songs:
            st.subheader("🎶 Recommended Spotify Tracks")
            for song in songs:
                col_img, col_info = st.columns([1, 3])
                with col_img:
                    if song["image"]:
                        st.image(song["image"], width=110)
                with col_info:
                    st.markdown(f"**🎵 {song['name']}** — *{song['artist']}*")
                    st.markdown(f"[Open in Spotify ↗]({song['url']})")
                # Embedded Spotify Web Player
                embed_url = f"https://open.spotify.com/embed/track/{song['id']}?utm_source=generator&theme=0"
                st.components.v1.iframe(embed_url, height=80)
                st.write("---")
        else:
            st.warning("No songs found")
