# Hybrid Movie Recommendation System

This project implements a **hybrid movie recommender** using **Content-Based Filtering (TF-IDF)** and **Collaborative Filtering (SVD)** with the MovieLens 100K dataset.

## Features
- Recommend personalized movies for users
- Weighted hybrid score (70% SVD, 30% content)
- Web interface via Streamlit
- Visualize top recommendations

## Requirements
- Python 3.8+
- Libraries:
  - pandas, numpy, scikit-learn, matplotlib, surprise, streamlit

Install dependencies via:
```bash
pip install -r requirements.txt