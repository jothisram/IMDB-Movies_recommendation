🎬 IMDb 2024 Movie Recommendation System

A content-based movie recommendation system built using Natural Language Processing (NLP) and Cosine Similarity to suggest similar movies based on genres, overview, keywords, and cast information.

This project demonstrates real-world implementation of text vectorization, similarity metrics, and deployment using Streamlit.

🚀 Project Overview

With thousands of movies released every year, discovering similar content becomes challenging.

This system analyzes IMDb 2024 movie metadata, processes textual features using NLP techniques, and recommends movies based on similarity scores computed through Cosine Similarity.

The result is a fast and intelligent content-based recommendation engine.

🧠 How It Works

Combine relevant textual features (genres, overview, keywords, cast).

Perform NLP preprocessing:

Tokenization

Stopword removal

Lemmatization

Convert text to numerical vectors using TF-IDF Vectorization

Compute similarity scores using Cosine Similarity

Recommend Top-N similar movies

🛠️ Tech Stack

Python

Pandas

NumPy

Scikit-learn

NLTK

Streamlit

📂 Project Structure
IMDB-Movies_recommendation/
│
├── app.py
├── imdb_scraper.py
├── imdb.ipynb
├── requirements.txt
├── README.md
└── .gitignore
▶️ How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/jothisram/IMDB-Movies_recommendation.git
cd IMDB-Movies_recommendation
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
streamlit run app.py
📊 Key Highlights

Content-Based Recommendation Engine

Real-world NLP Implementation

TF-IDF Vector Space Model

Cosine Similarity for Ranking

Streamlit UI for Interactive Experience

🎯 Learning Outcomes

Hands-on implementation of NLP pipeline

Understanding Vector Space Models

Working with similarity metrics

Building deployable ML applications

📌 Future Improvements

Add collaborative filtering

Deploy on Streamlit Cloud

Add user-based personalization

Improve model scalability

👨‍💻 Author

Jothisram R
Computer Science Graduate
Aspiring Data / ML Engineer
