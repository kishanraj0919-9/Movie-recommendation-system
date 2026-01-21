import streamlit as st
import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Content-Based Movie Recommender using TMDB Dataset")

# ---------- Load Data ----------
@st.cache_data
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies.rename(columns={'id': 'movie_id'}, inplace=True)

    movies = movies.merge(credits, on='movie_id')

    # FIX for title error
    if 'original_title' in movies.columns:
        movies.rename(columns={'original_title': 'title'}, inplace=True)

    return movies


movies = load_data()

# ---------- Data Processing ----------
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert_cast(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

for col in ['genres','keywords','cast','crew']:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)

movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

final_df = movies[['movie_id','title','tags']]

# ---------- Vectorization ----------
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(final_df['tags'])

similarity = cosine_similarity(vectors)

# ---------- Recommendation Function ----------
def recommend(movie):
    if movie not in final_df['title'].values:
        return []

    index = final_df[final_df['title'] == movie].index[0]

    distances = sorted(
        list(enumerate(similarity[index])),
        key=lambda x: x[1],
        reverse=True
    )

    return [final_df.iloc[i[0]].title for i in distances[1:6]]

# ---------- Streamlit UI ----------
movie_list = sorted(final_df['title'].values)
selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    if recommendations:
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write("👉", movie)
    else:
        st.error("Movie not found!")
#








# import pickle
# import streamlit as st
# import requests
#
#
# # ---------- Poster fetch function with caching & safe fallback ----------
# @st.cache_data(show_spinner=False)
# def fetch_poster(movie_title):
#     # Clean the title for OMDb query
#     movie_title = movie_title.replace("’", "'").replace(".", "").replace("&", "and")
#
#     api_key = st.secrets["OMDB_API_KEY"]
#     url = "https://www.omdbapi.com/"
#     params = {
#         "t": movie_title,
#         "apikey": api_key
#     }
#
#     try:
#         response = requests.get(url, params=params, timeout=5)
#         data = response.json()
#
#         poster_url = data.get("Poster")
#         if poster_url and poster_url != "N/A":
#             return poster_url
#         else:
#             # fallback if poster not found
#             return "https://via.placeholder.com/500x750?text=No+Image"
#
#     except requests.exceptions.RequestException:
#         # fallback if network/API fails
#         return "https://via.placeholder.com/500x750?text=Error"
#
#
# # ---------- Recommendation function ----------
# def recommend(movie):
#     if movie not in movies['title'].values:
#         return [], []
#
#     index = movies[movies['title'] == movie].index[0]
#     distances = sorted(
#         list(enumerate(similarity[index])),
#         key=lambda x: x[1],
#         reverse=True
#     )
#
#     recommended_movie_names = []
#     recommended_movie_posters = []
#
#     for i in distances[1:6]:
#         movie_title = movies.iloc[i[0]].title
#         recommended_movie_names.append(movie_title)
#         recommended_movie_posters.append(fetch_poster(movie_title))
#
#     return recommended_movie_names, recommended_movie_posters
#
#
# # ---------- Streamlit UI ----------
# st.set_page_config(page_title="Movie Recommender", layout="wide")
# st.title("🎬 Movie Recommendation System")
# st.write("Content-Based Movie Recommender with OMDb Posters")
#
# # ---------- Load data ----------
# movies = pickle.load(open('movie_list.pkl', 'rb'))
# similarity = pickle.load(open('similarity.pkl', 'rb'))
#
# movie_list = movies['title'].values
# selected_movie = st.selectbox(
#     "Type or select a movie from the dropdown",
#     movie_list
# )
#
# # ---------- Show recommendations ----------
# if st.button('Show Recommendation'):
#     recommended_names, recommended_posters = recommend(selected_movie)
#
#     if recommended_names:
#         cols = st.columns(5)  # 5-column layout
#         for i in range(5):
#             with cols[i]:
#                 st.text(recommended_names[i])
#                 st.image(recommended_posters[i])
#     else:
#         st.error("Movie not found!")
