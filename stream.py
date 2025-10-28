import streamlit as st
from typing import List
import numpy as np
import pandas as pd
import plotly.express as px

# Local imports (safe try/except to prevent blocking)
try:
    from news_fetcher import fetch_live_news
    from recommender import NewsRecommender
except Exception as e:
    fetch_live_news = None
    NewsRecommender = None
    st.error(f"Import error: {e}")

st.set_page_config(page_title="AI News Recommender", page_icon="📰", layout="wide")
st.title("📰 AI News Recommender")

@st.cache_data(ttl=300)
def get_articles(query: str, max_articles: int = 30):
    if fetch_live_news is None:
        return [
            {"title": f"Sample Article {i+1}", "description": "Demo description", "link": "https://example.com"}
            for i in range(max_articles)
        ]
    try:
        return fetch_live_news(query=query, max_articles=max_articles)
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return [{"title": "Fallback Article", "description": "News fetching failed", "link": "https://example.com/fallback"}]

@st.cache_resource
def get_recommender():
    if NewsRecommender is None:
        return None
    return NewsRecommender()

# --- Search box ---
query = st.text_input("🔎 Search for news", value="AI")
search_btn = st.button("Search")

articles = []
if search_btn and query:
    with st.spinner("Fetching and recommending articles..."):
        articles = get_articles(query, max_articles=16)

# --- Display recommendations ---
if articles:
    recommender = get_recommender()
    if recommender is not None:
        try:
            recommender.build_index(articles)
            recs = recommender.recommend(query, top_k=16)
        except Exception as e:
            st.error(f"Recommendation error: {e}")
            recs = articles[:16]
    else:
        recs = articles[:16]

    st.subheader("Recommended Articles")
    cols = st.columns(4)
    for i, article in enumerate(recs):
        with cols[i % 4]:
            st.markdown(f"**{article.get('title', 'Untitled')}**")
            if article.get('description'):
                st.caption(article['description'])
            if article.get('link'):
                st.markdown(f"[Read more]({article['link']})")
            st.write("---")

    # --- Plot of recommendation scores ---
    if recommender:
        sim_scores = []
        titles = []
        recs_for_plot = recommender.recommend(query, top_k=16, return_scores=True)
        for r in recs_for_plot:
            titles.append(r['title'])
            sim_scores.append(r['score'])  # assuming recommender returns score

        df = pd.DataFrame({"Article": titles, "Score": sim_scores})
        fig = px.bar(df, x="Score", y="Article", orientation="h", text="Score",
                     title=f"Recommendation Scores for '{query}'")
        st.plotly_chart(fig, use_container_width=True)

    # --- Evaluation section ---
    st.subheader("Evaluation Metrics")
    hits, precision_scores, ndcg_scores, rr_scores, ap_scores = [], [], [], [], []
    retrieved_all = []

    for art in articles:
        q = art.get('title', '')
        recs_eval = recommender.recommend(q, top_k=5) if recommender else []
        retrieved_titles = [r.get('title') for r in recs_eval]
        retrieved_all.extend(retrieved_titles)

        hit = int(art.get('title') in retrieved_titles)
        hits.append(hit)
        precision_scores.append(hit / 5 if 5 else 0)

        if hit:
            rank = retrieved_titles.index(art.get('title')) + 1
            ndcg = 1 / np.log2(rank + 1)
            rr = 1 / rank
        else:
            ndcg = 0.0
            rr = 0.0
        ndcg_scores.append(ndcg)
        rr_scores.append(rr)

        rels = [1 if title == art.get('title') else 0 for title in retrieved_titles]
        if sum(rels) > 0:
            precisions = [sum(rels[:i+1])/(i+1) for i in range(len(rels)) if rels[i] == 1]
            ap = np.mean(precisions)
        else:
            ap = 0.0
        ap_scores.append(ap)

    coverage = len(set(retrieved_all)) / len(articles) if articles else 0.0

    st.json({
        'hit_ratio@5': float(np.mean(hits)) if hits else 0.0,
        'precision@5': float(np.mean(precision_scores)) if precision_scores else 0.0,
        'ndcg@5': float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        'mrr': float(np.mean(rr_scores)) if rr_scores else 0.0,
        'map': float(np.mean(ap_scores)) if ap_scores else 0.0,
        'coverage': coverage
    })
else:
    st.info("Enter a query and click Search to get recommendations.")

st.caption("Built with sentence-transformers + faiss. Ensure dependencies are installed.")
