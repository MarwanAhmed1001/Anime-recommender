# 🎌 Anime Recommender System

A content-based anime recommender system that helps users discover new anime based on their preferences — supports English, Arabic, and Japanese search!

## 🌟 Features
- **Smart Search** — Search in English, Arabic, or Japanese
- **Cold-Start Quiz** — New to anime? Answer 3 questions and get perfect recommendations
- **Hybrid Scoring** — Combines genre similarity, rating, and popularity
- **Franchise Diversity** — Won't flood you with sequels
- **Anime Images** — Visual cards for every recommendation

## 🧠 How It Works
1. **TF-IDF Vectorization** — Converts anime genres into numerical vectors
2. **Cosine Similarity** — Measures similarity between any two anime
3. **Hybrid Scoring Formula:**
   - 60% Genre Similarity
   - 25% Normalized Rating
   - 15% Normalized Popularity
4. **Franchise Filter** — Limits each franchise to max 2 appearances
5. **International Search** — Translates Arabic/English via Google Translate then matches using Jikan API

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Scikit-learn | TF-IDF & Cosine Similarity |
| Streamlit | Web UI |
| Jikan API | Anime images & name matching |
| Deep Translator | Arabic/English translation |

## 📦 Dataset
- **Source:** [MyAnimeList Dataset on Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
- **Size:** 12,232 anime titles
- **Features:** name, genre, type, episodes, rating, members

## 🚀 Run Locally

1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/anime-recommender.git
cd anime-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
python -m streamlit run app.py
```
## 👨‍💻 Author
**Marwan** 

Built from scratch as a data science portfolio project.
Recommender engine built using content-based filtering with hybrid scoring.

- GitHub: [MarwanAhmed1001](https://github.com/MarwanAhmed1001)
- LinkedIn: [Marwan Ahmed](https://www.linkedin.com/in/marwan-ahmed-173549374/)

