import streamlit as st
import pandas as pd
import pickle
from surprise import SVD
import requests
import re

# --- Style ---
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(to right, #1f1c2c, #928dab);
        color: white;
    }
    .css-10trblm, .css-1v0mbdj, .stMarkdown, .stTextInput, .stSlider, .stSelectbox {
        color: white !important;
    }
    .overview-text {
        font-size: 13px;
        color: #ddd;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #FFD700;'> Hybrid Movie Recommender 🎬</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Combining collaborative & content filtering to find your next watch 🎥</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Load Data ---
@st.cache_data
def load_data():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', names=[
        'movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ])
    return ratings, movies

@st.cache_resource
def load_model():
    with open("svd_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_cbf():
    with open("cbf_matrix.pkl", "rb") as f:
        return pickle.load(f)

# --- Genre extractor ---
def extract_genres(row):
    genres = []
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                  'Sci-Fi', 'Thriller', 'War', 'Western']
    for genre in genre_cols:
        if row[genre] == 1:
            genres.append(genre)
    return ', '.join(genres)

# --- Clean title ---
def clean_title(title):
    return re.sub(r"\s*\(\d{4}\)$", "", title).strip()

# --- TMDb Poster/Overview ---
TMDB_API_KEY = "86446ee93ac1b2aa226319f6efbcfa33"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_data(ttl=3600)
def get_movie_info_tmdb_from_title(title):
    title = clean_title(title)
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    response = requests.get(search_url, params=params)
    if response.status_code != 200:
        return None, None
    results = response.json().get("results", [])
    if not results:
        return None, None
    movie = results[0]
    poster_path = movie.get('poster_path')
    overview = movie.get('overview', '')
    poster_url = TMDB_IMAGE_BASE_URL + poster_path if poster_path else "https://via.placeholder.com/300x450?text=No+Image"
    return poster_url, overview

# --- Hybrid Recommender ---
def get_recommendations(user_id, algo, cbf_df, movies, ratings, top_n):
    user_rated = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    candidates = [m for m in movies['movieId'] if m not in user_rated]
    recs = []

    for movie_id in candidates:
        try:
            pred = algo.predict(user_id, movie_id).est
            cbf_score = cbf_df.loc[movie_id].drop(user_rated, errors='ignore').mean() if movie_id in cbf_df.index else 0
            hybrid_score = 0.7 * pred + 0.3 * cbf_score
            recs.append((movie_id, hybrid_score))
        except:
            continue

    recs = sorted(recs, key=lambda x: x[1], reverse=True)[:top_n]
    recommended = movies[movies['movieId'].isin([r[0] for r in recs])].copy()
    recommended['Hybrid Score'] = [round(r[1], 2) for r in recs]
    return recommended

# --- Load Everything ---
with st.spinner("Loading..."):
    ratings, movies = load_data()
    algo = load_model()
    cbf_df = load_cbf()

# --- Sidebar ---
st.sidebar.header(" Customize Recommendations")
user_id = st.sidebar.number_input("Enter User ID (1–943)", min_value=1, max_value=943, step=1)
num_recommendations = st.sidebar.slider("Number of recommendations", 1, 10, 5)

if st.sidebar.button(" Show Recommendations"):
    with st.spinner("Fetching your recommendations..."):
        results = get_recommendations(user_id, algo, cbf_df, movies, ratings, top_n=num_recommendations)

        for idx, row in results.iterrows():
            title = row['title']
            genres = extract_genres(row)
            score = row['Hybrid Score']
            poster_url, description = get_movie_info_tmdb_from_title(title)
            if not description:
                description = f"A {genres} movie." if genres else "No description available."

            st.markdown(f"### {title}")
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(poster_url, use_container_width=True)
            with cols[1]:
                st.markdown(f"**Genres:** {genres}")
                st.markdown(f"**Score:**  {score}")
                st.markdown(f"<div class='overview-text'>{description}</div>", unsafe_allow_html=True)
            st.markdown("---")
