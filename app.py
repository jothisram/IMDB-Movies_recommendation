import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass

download_nltk_data()

# IMDb Color Theme
IMDB_YELLOW = "#F5C518"
IMDB_BLACK = "#121212"
IMDB_DARK_GRAY = "#1F1F1F"
IMDB_LIGHT_GRAY = "#2E2E2E"

# Custom CSS for IMDb theme
st.set_page_config(
    page_title="IMDb Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
<style>
    /* Main theme colors */
    .stApp {{
        background-color: {IMDB_BLACK};
    }}

    /* Headers */
    h1, h2, h3 {{
        color: {IMDB_YELLOW} !important;
        font-family: 'Arial', sans-serif;
    }}

    /* Text */
    p, div, span, label {{
        color: #FFFFFF !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {IMDB_DARK_GRAY};
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {IMDB_YELLOW};
        color: {IMDB_BLACK};
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 16px;
    }}

    .stButton > button:hover {{
        background-color: #FFD700;
        color: {IMDB_BLACK};
    }}

    /* Input boxes */
    .stTextInput > div > div > input {{
        background-color: {IMDB_LIGHT_GRAY};
        color: white;
        border: 2px solid {IMDB_YELLOW};
    }}

    .stTextArea > div > div > textarea {{
        background-color: {IMDB_LIGHT_GRAY};
        color: white;
        border: 2px solid {IMDB_YELLOW};
    }}

    /* Selectbox */
    .stSelectbox > div > div {{
        background-color: {IMDB_LIGHT_GRAY};
        color: white;
    }}

    /* Cards */
    .movie-card {{
        background-color: {IMDB_DARK_GRAY};
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {IMDB_YELLOW};
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }}

    .movie-title {{
        color: {IMDB_YELLOW};
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }}

    .movie-storyline {{
        color: #CCCCCC;
        font-size: 14px;
        line-height: 1.6;
    }}

    .similarity-score {{
        background-color: {IMDB_YELLOW};
        color: {IMDB_BLACK};
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
    }}

    /* Statistics boxes */
    .stat-box {{
        background-color: {IMDB_DARK_GRAY};
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 2px solid {IMDB_YELLOW};
    }}

    .stat-number {{
        color: {IMDB_YELLOW};
        font-size: 32px;
        font-weight: bold;
    }}

    .stat-label {{
        color: #CCCCCC;
        font-size: 14px;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: {IMDB_DARK_GRAY};
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {IMDB_LIGHT_GRAY};
        color: white;
        border-radius: 5px;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {IMDB_YELLOW};
        color: {IMDB_BLACK};
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {IMDB_DARK_GRAY};
        color: {IMDB_YELLOW} !important;
        font-weight: bold;
    }}

    /* Success/Error messages */
    .stSuccess {{
        background-color: {IMDB_DARK_GRAY};
        border-left: 5px solid #00FF00;
    }}

    .stError {{
        background-color: {IMDB_DARK_GRAY};
        border-left: 5px solid #FF0000;
    }}
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load the movie dataset"""
    try:
        df = pd.read_csv('imdb_2024_movies.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load pre-trained models"""
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('cosine_similarity.pkl', 'rb') as f:
            cosine_sim = pickle.load(f)
        return tfidf, cosine_sim
    except:
        return None, None

# Text preprocessing function
@st.cache_data
def preprocess_text(text):
    """Preprocess text for NLP"""
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        cleaned_tokens = [
            lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in stop_words and len(token) > 2
        ]
        return ' '.join(cleaned_tokens)
    except:
        return text

# Recommendation functions
def get_recommendations_by_title(movie_title, df, cosine_sim, top_n=5):
    """Get recommendations based on movie title"""
    try:
        idx = df[df['Movie Title'].str.lower() == movie_title.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]

        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]

        recommendations = pd.DataFrame({
            'Rank': range(1, top_n+1),
            'Movie Title': df['Movie Title'].iloc[movie_indices].values,
            'Storyline': df['Storyline'].iloc[movie_indices].values,
            'Similarity Score': similarity_scores
        })

        return recommendations, df['Storyline'].iloc[idx]
    except IndexError:
        return None, None

def get_recommendations_by_storyline(user_storyline, df, tfidf, top_n=5):
    """Get recommendations based on custom storyline"""
    try:
        # Preprocess input
        cleaned_input = preprocess_text(user_storyline)

        # Transform to TF-IDF vector
        input_vector = tfidf.transform([cleaned_input])

        # Load TF-IDF matrix for all movies
        cleaned_storylines = df['Storyline'].apply(preprocess_text)
        tfidf_matrix = tfidf.transform(cleaned_storylines)

        # Calculate similarity
        similarities = cosine_similarity(input_vector, tfidf_matrix)[0]

        # Get top N
        top_indices = similarities.argsort()[-top_n:][::-1]

        recommendations = pd.DataFrame({
            'Rank': range(1, top_n+1),
            'Movie Title': df['Movie Title'].iloc[top_indices].values,
            'Storyline': df['Storyline'].iloc[top_indices].values,
            'Similarity Score': [similarities[i] for i in top_indices]
        })

        return recommendations
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Main app
def main():
    # Header
    st.markdown(f"""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='font-size: 48px; margin-bottom: 0;'>
            🎬 IMDb Movie Recommender
        </h1>
        <p style='color: {IMDB_YELLOW}; font-size: 18px;'>
            Discover Your Next Favorite Movie Using AI-Powered Recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data()
    tfidf, cosine_sim = load_models()

    if df is None:
        st.error("❌ Unable to load dataset. Please check if 'imdb_2024_movies.csv' exists.")
        return

    # Sidebar
    with st.sidebar:
        st.markdown(f"<h2 style='text-align: center;'>⚙️ Settings</h2>", unsafe_allow_html=True)

        st.markdown("---")

        # Number of recommendations
        top_n = st.slider("Number of Recommendations", 3, 10, 5)

        st.markdown("---")

        # Dataset statistics
        st.markdown(f"<h3>📊 Dataset Info</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-number'>{len(df)}</div>
                <div class='stat-label'>Total Movies</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-number'>2024</div>
                <div class='stat-label'>Release Year</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # About section
        with st.expander("ℹ️ About This App"):
            st.markdown("""
            This recommendation system uses:
            - **NLP**: Text preprocessing & analysis
            - **TF-IDF**: Feature extraction
            - **Cosine Similarity**: Similarity measurement

            **How it works:**
            1. Choose recommendation method
            2. Enter movie title or storyline
            3. Get top similar movies!
            """)

    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Find by Title", "✍️ Find by Storyline", "📚 Browse All Movies"])

    # Tab 1: Recommendations by Title
    with tab1:
        st.markdown("<h2>Find Similar Movies by Title</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        with col1:
            selected_movie = st.selectbox(
                "Select a movie:",
                options=df['Movie Title'].tolist(),
                key='title_select'
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_button = st.button("🔍 Find Similar Movies", key='title_search')

        if search_button or selected_movie:
            if cosine_sim is not None:
                recommendations, original_storyline = get_recommendations_by_title(
                    selected_movie, df, cosine_sim, top_n
                )

                if recommendations is not None:
                    # Display original movie
                    st.markdown("<h3>📽️ Selected Movie</h3>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='movie-card'>
                        <div class='movie-title'>{selected_movie}</div>
                        <div class='movie-storyline'>{original_storyline}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Display recommendations
                    st.markdown(f"<h3>🎬 Top {top_n} Similar Movies</h3>", unsafe_allow_html=True)

                    for _, row in recommendations.iterrows():
                        st.markdown(f"""
                        <div class='movie-card'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div class='movie-title'>#{row['Rank']} - {row['Movie Title']}</div>
                                <div class='similarity-score'>⭐ {row['Similarity Score']:.2%}</div>
                            </div>
                            <div class='movie-storyline'>{row['Storyline']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("Movie not found in database.")
            else:
                st.warning("⚠️ Models not loaded. Please run imdb.ipynb first to generate model files.")

    # Tab 2: Recommendations by Storyline
    with tab2:
        st.markdown("<h2>Find Movies by Custom Storyline</h2>", unsafe_allow_html=True)

        st.markdown("""
        <p style='color: #CCCCCC; margin-bottom: 20px;'>
        Describe a movie plot, and we'll find similar movies for you!
        </p>
        """, unsafe_allow_html=True)

        user_storyline = st.text_area(
            "Enter your storyline:",
            height=150,
            placeholder="Example: A young wizard begins his journey at a magical school where he makes friends and enemies...",
            key='storyline_input'
        )

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            search_storyline_button = st.button("🔍 Find Movies", key='storyline_search')
        with col2:
            clear_button = st.button("🗑️ Clear", key='clear_storyline')

        if clear_button:
            st.rerun()

        if search_storyline_button and user_storyline:
            if tfidf is not None:
                with st.spinner("Analyzing storyline and finding matches..."):
                    recommendations = get_recommendations_by_storyline(
                        user_storyline, df, tfidf, top_n
                    )

                if recommendations is not None and len(recommendations) > 0:
                    st.success(f"✅ Found {len(recommendations)} similar movies!")

                    st.markdown(f"<h3>🎬 Top {top_n} Matching Movies</h3>", unsafe_allow_html=True)

                    for _, row in recommendations.iterrows():
                        st.markdown(f"""
                        <div class='movie-card'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div class='movie-title'>#{row['Rank']} - {row['Movie Title']}</div>
                                <div class='similarity-score'>⭐ {row['Similarity Score']:.2%}</div>
                            </div>
                            <div class='movie-storyline'>{row['Storyline']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("No recommendations found. Please try a different storyline.")
            else:
                st.warning("⚠️ Models not loaded. Please run imdb.ipynb first to generate model files.")
        elif search_storyline_button:
            st.warning("⚠️ Please enter a storyline first!")

    # Tab 3: Browse All Movies
    with tab3:
        st.markdown("<h2>Browse All Movies in Database</h2>", unsafe_allow_html=True)

        # Search box
        search_query = st.text_input("🔍 Search movies:", placeholder="Enter movie title or keyword...")

        # Filter dataframe
        if search_query:
            filtered_df = df[
                df['Movie Title'].str.contains(search_query, case=False) | 
                df['Storyline'].str.contains(search_query, case=False)
            ]
        else:
            filtered_df = df

        st.markdown(f"<p style='color: {IMDB_YELLOW};'>Showing {len(filtered_df)} movies</p>", unsafe_allow_html=True)

        # Display movies in cards
        for _, row in filtered_df.iterrows():
            st.markdown(f"""
            <div class='movie-card'>
                <div class='movie-title'>{row['Movie Title']}</div>
                <div class='movie-storyline'>{row['Storyline']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: {IMDB_DARK_GRAY}; border-radius: 10px;'>
        <p style='color: {IMDB_YELLOW}; font-size: 16px; margin: 0;'>
            🎬 Developed with ❤️ by Professional Data Scientist | 
            Powered by NLP & Machine Learning
        </p>
        <p style='color: #CCCCCC; font-size: 12px; margin-top: 10px;'>
            Dataset: IMDb 2024 Movies | Technology: Python, Streamlit, Scikit-learn, NLTK
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
