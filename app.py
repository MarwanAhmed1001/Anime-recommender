import streamlit as st
import pandas as pd
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")

st.markdown("""
<style>
    .anime-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid #e94560;
    }
    .anime-title { color: #ffffff; font-size: 16px; font-weight: bold; margin-bottom: 4px; }
    .anime-genre { color: #a0a0b0; font-size: 12px; margin-bottom: 6px; }
    .anime-badge { background: #e94560; color: white; padding: 2px 8px; border-radius: 20px; font-size: 11px; margin-right: 4px; }
    .rating-badge { background: #f5a623; color: #000; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: bold; }
    .stTextInput input { background-color: #1a1a2e !important; color: white !important; border: 2px solid #e94560 !important; border-radius: 12px !important; font-size: 16px !important; padding: 12px !important; }
    .stButton button { background: linear-gradient(135deg, #e94560, #c23152) !important; color: white !important; border: none !important; border-radius: 12px !important; font-size: 16px !important; font-weight: bold !important; width: 100% !important; }
    h1, h2, h3 { color: #ffffff !important; }
    .stRadio label { color: #ffffff !important; }
    .stSelectbox label { color: #ffffff !important; }
    .stTextInput label { color: #ffffff !important; }
    .section-title { color: #e94560; font-size: 22px; font-weight: bold; margin: 20px 0 10px 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Download dataset directly from Kaggle
    url = "https://raw.githubusercontent.com/MarwanAhmed1001/anime-recommender/main/anime.csv"
    df = pd.read_csv(url)
    
    df = df.dropna(subset=["genre"])
    df["type"] = df["type"].fillna("Unknown")
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
    df["episodes"] = df["episodes"].fillna(df["episodes"].median())
    df = df.reset_index(drop=True)
    df["genre_clean"] = df["genre"].str.replace(", ", " ")
    
    scaler = MinMaxScaler()
    df["rating_norm"] = scaler.fit_transform(df[["rating"]])
    df["members_norm"] = scaler.fit_transform(df[["members"]])
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["genre_clean"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df["name"]).drop_duplicates()
    
    return df, cosine_sim, indices

df, cosine_sim, indices = load_data()

@st.cache_data
def get_anime_image(anime_name):
    try:
        time.sleep(0.4)
        response = requests.get(
            "https://api.jikan.moe/v4/anime?q=" + anime_name + "&limit=1",
            timeout=5
        )
        data = response.json()
        if data["data"]:
            img = data["data"][0]["images"]["jpg"]["large_image_url"]
            if img:
                return img
        short_name = " ".join(anime_name.split()[:2])
        time.sleep(0.4)
        response = requests.get(
            "https://api.jikan.moe/v4/anime?q=" + short_name + "&limit=1",
            timeout=5
        )
        data = response.json()
        if data["data"]:
            img = data["data"][0]["images"]["jpg"]["large_image_url"]
            if img:
                return img
    except:
        pass
    return "https://placehold.co/200x280/1a1a2e/e94560?text=No+Image"

def get_franchise(name):
    name_lower = name.lower()
    known = ["naruto", "dragon ball", "bleach", "one piece", "pokemon", "saint seiya", "boruto", "gintama"]
    for f in known:
        if f in name_lower:
            return f
    return name.split()[0].lower().strip(":")

def recommend_final(anime_name, num=10):
    if anime_name not in indices:
        return None
    idx = indices[anime_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    scores_df = pd.DataFrame(sim_scores, columns=["idx", "genre_sim"])
    scores_df["rating_norm"] = df["rating_norm"].values
    scores_df["members_norm"] = df["members_norm"].values
    scores_df["hybrid_score"] = (
        0.6 * scores_df["genre_sim"] +
        0.25 * scores_df["rating_norm"] +
        0.15 * scores_df["members_norm"]
    )
    scores_df = scores_df.sort_values("hybrid_score", ascending=False)
    scores_df = scores_df[scores_df["idx"] != idx]
    results, franchise_count = [], {}
    for _, row in scores_df.iterrows():
        name = df["name"].iloc[int(row["idx"])]
        franchise = get_franchise(name)
        franchise_count[franchise] = franchise_count.get(franchise, 0) + 1
        if franchise_count[franchise] <= 2:
            results.append(row)
        if len(results) == num:
            break
    final_df = pd.DataFrame(results)
    result = df[["name", "genre", "rating", "type"]].iloc[final_df["idx"].astype(int).values].copy()
    result["score"] = final_df["hybrid_score"].values.round(3)
    return result

def smart_search(query):
    try:
        translated = GoogleTranslator(source="auto", target="english").translate(query)
    except:
        translated = query
    try:
        response = requests.get(
            "https://api.jikan.moe/v4/anime?q=" + translated + "&limit=1",
            timeout=5
        )
        data = response.json()
        if not data["data"]:
            return None, None, None
        anime = data["data"][0]
        english_name = anime["title"]
        japanese_name = anime["title_japanese"]
        image_url = anime["images"]["jpg"]["large_image_url"]
        match = df[df["name"] == japanese_name]
        if match.empty:
            match = df[df["name"] == english_name]
        if match.empty:
            base = english_name.split(":")[0].strip()
            match = df[df["name"].str.lower() == base.lower()]
        if match.empty:
            return None, english_name, image_url
        return match.iloc[0]["name"], english_name, image_url
    except:
        return None, None, None

def render_card(name, genre, rating, anime_type):
    st.markdown(
        "<div class=\"anime-card\">"
        "<div class=\"anime-title\">" + name + "</div>"
        "<div class=\"anime-genre\">" + genre + "</div>"
        "<span class=\"anime-badge\">" + anime_type + "</span>"
        "<span class=\"rating-badge\">⭐ " + str(rating) + "</span>"
        "</div>",
        unsafe_allow_html=True
    )

mood_to_genres = {
    "🌑 Dark & Serious": ["Psychological", "Thriller", "Horror", "Drama", "Mystery", "Supernatural"],
    "😄 Fun & Lighthearted": ["Comedy", "Parody", "School", "Slice of Life", "Kids"],
    "⚔️ Action-Packed": ["Action", "Adventure", "Martial Arts", "Super Power", "Shounen", "Military"],
    "💕 Romantic": ["Romance", "Drama", "Shoujo", "School"],
    "✨ Fantasy & Magic": ["Fantasy", "Magic", "Demons", "Supernatural", "Adventure"],
}

length_to_range = {
    "🎬 Short (1-13 eps)": (1, 13),
    "📺 Medium (14-50 eps)": (14, 50),
    "🎭 Long (50+ eps)": (51, 10000),
    "🔀 Any length": (0, 10000),
}

st.markdown("<h1 style=\"text-align:center; color:#e94560;\">🎌 Anime Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style=\"text-align:center; color:#a0a0b0; font-size:18px;\">Discover your next favorite anime in seconds</p>", unsafe_allow_html=True)
st.markdown("---")

mode = st.radio("", ["🎯 I know an anime I like", "🆕 I am new — take the quiz"], horizontal=True)
st.markdown("---")

if mode == "🎯 I know an anime I like":
    query = st.text_input("", placeholder="Search in English, Arabic, or Japanese... e.g. هجوم العمالقة")
    if st.button("🔍 Get Recommendations") and query:
        with st.spinner("Searching..."):
            matched_name, english_name, search_image = smart_search(query)
        if not matched_name:
            st.warning("Could not find this anime. Try the quiz instead!")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                if search_image:
                    st.image(search_image, width=180)
            with col2:
                st.markdown("<div class=\"section-title\">Found: " + matched_name + "</div>", unsafe_allow_html=True)
                st.markdown("<p style=\"color:#a0a0b0;\">Here are anime similar to your pick:</p>", unsafe_allow_html=True)
            st.markdown("---")
            results = recommend_final(matched_name)
            if results is not None:
                cols = st.columns(2)
                for i, (_, row) in enumerate(results.iterrows()):
                    image_url = get_anime_image(row["name"])
                    with cols[i % 2]:
                        img_col, info_col = st.columns([1, 2])
                        with img_col:
                            st.image(image_url, width=100)
                        with info_col:
                            render_card(row["name"], row["genre"], row["rating"], row["type"])

else:
    st.markdown("<div class=\"section-title\">Tell us your preferences 🎯</div>", unsafe_allow_html=True)
    mood = st.selectbox("What is your mood right now?", list(mood_to_genres.keys()))
    length = st.selectbox("How long do you want it?", list(length_to_range.keys()))
    anime_type = st.selectbox("What type?", ["Any", "TV", "Movie", "OVA", "Special"])

    if st.button("🚀 Find My Perfect Anime!"):
        target_genres = mood_to_genres[mood]
        min_ep, max_ep = length_to_range[length]
        filtered = df.copy()
        filtered = filtered[(filtered["episodes"] >= min_ep) & (filtered["episodes"] <= max_ep)]
        if anime_type != "Any":
            filtered = filtered[filtered["type"] == anime_type]
        genre_mask = filtered["genre"].apply(lambda x: any(g in x for g in target_genres))
        filtered = filtered[genre_mask].copy()
        filtered["hybrid_score"] = (0.6 * filtered["rating_norm"] + 0.4 * filtered["members_norm"])
        results = filtered.sort_values("hybrid_score", ascending=False).head(10)
        st.markdown("---")
        st.markdown("<div class=\"section-title\">Your perfect anime picks 🎌</div>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (_, row) in enumerate(results.iterrows()):
            image_url = get_anime_image(row["name"])
            with cols[i % 2]:
                img_col, info_col = st.columns([1, 2])
                with img_col:
                    st.image(image_url, width=100)
                with info_col:
                    render_card(row["name"], row["genre"], row["rating"], row["type"])