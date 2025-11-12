# ------------------------------------------------------------
# Streamlit UI + Backend Logic (integrated version)
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import logging

# Import recommender algorithms
from model import (
    collaborative_filtering,
    content_based_filtering,
    hybrid_recommendation,
    MultiModalModel
)

# ------------------ CONFIGURATION ------------------
st.set_page_config(page_title="Product Recommender", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit-app")

st.title("🛍️ Product Recommendation System — Streamlit + Backend")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    files = {
        "users": "users_expanded.csv",
        "products": "products_expanded.csv",
        "product_images": "product_images_expanded.csv",
        "purchases": "purchases_expanded.csv",
        "browsing_history": "browsing_history_expanded.csv"
    }
    dfs = {}
    for key, path in files.items():
        if not os.path.exists(path):
            st.warning(f"⚠️ Missing file: {path}")
            dfs[key] = pd.DataFrame()
        else:
            dfs[key] = pd.read_csv(path)
    return dfs["users"], dfs["products"], dfs["product_images"], dfs["purchases"], dfs["browsing_history"]

users, products, product_images, purchases, browsing_history = load_data()

# ------------------ SIDEBAR CONTROLS ------------------
st.sidebar.header("⚙️ Configuration")

if users.empty:
    st.error("Missing `users_expanded.csv` — cannot run demo.")
    st.stop()

user_ids = users['user_id'].dropna().astype(int).sort_values().tolist()
default_user = user_ids[0] if user_ids else 1

user_id = st.sidebar.number_input("User ID", min_value=1, value=default_user, step=1)
algorithms = ["collaborative", "content-based", "hybrid", "multi-modal"]
algorithm = st.sidebar.selectbox("Select Algorithm", algorithms, index=0)
top_k = st.sidebar.slider("Top-K Recommendations", 1, 50, 10)
run_button = st.sidebar.button("🚀 Run Recommendation")

# ------------------ MAIN LOGIC ------------------
if run_button:
    try:
        if user_id not in users['user_id'].values:
            st.error("User ID not found in dataset.")
            st.stop()

        # --- GET USER INTERACTIONS ---
        purchased_ids = purchases[purchases['user_id'] == user_id]['product_id'].unique()
        browsed_ids = browsing_history[browsing_history['user_id'] == user_id]['product_id'].unique()

        interacted = products[products['product_id'].isin(np.union1d(purchased_ids, browsed_ids))].copy()
        if not interacted.empty:
            interacted['source'] = interacted['product_id'].apply(
                lambda x: "Purchased" if x in purchased_ids else "Browsed"
            )
            st.subheader("🧾 Interacted Products")
            st.dataframe(interacted)
        else:
            st.info("No prior interactions found for this user.")

        # --- CHOOSE ALGORITHM ---
        recs = pd.DataFrame()
        multimodal_ok = (
            'description' in products.columns
            and not product_images.empty
        )

        if algorithm == "collaborative":
            recs = collaborative_filtering(user_id, purchases, products)

        elif algorithm == "content-based":
            recs = content_based_filtering(user_id, purchases, browsing_history, products)

        elif algorithm == "hybrid":
            recs = hybrid_recommendation(user_id, purchases, browsing_history, products)

        elif algorithm == "multi-modal":
            if not multimodal_ok:
                st.warning("⚠️ Missing data (descriptions or images) for multi-modal model.")
                recs = pd.DataFrame()
            else:
                num_users = users['user_id'].nunique()
                num_products = products['product_id'].nunique()
                model = MultiModalModel(num_users, num_products)

                product_ids_tensor = torch.LongTensor(products['product_id'].values) - 1
                texts = products['description'].fillna("").tolist()

                with torch.no_grad():
                    outputs = model(
                        torch.LongTensor([user_id - 1]),
                        product_ids_tensor,
                        texts,
                        edge_index=None,
                        product_images_df=product_images
                    )

                scores = outputs.mean(dim=1).cpu().numpy()
                recs = products.copy()
                recs['score'] = scores
                recs['source'] = 'Multi-Modal'
        else:
            st.warning("Invalid algorithm selected.")
            recs = pd.DataFrame()

        # --- POST-PROCESSING ---
        if recs is None or recs.empty:
            st.info("No recommendations available for this user.")
        else:
            # Remove products user already interacted with
            recs = recs[
                ~recs['product_id'].isin(purchased_ids) &
                ~recs['product_id'].isin(browsed_ids)
            ].copy()

            if 'score' not in recs.columns:
                recs['score'] = 0.0

            recs = recs.sort_values('score', ascending=False).head(top_k)

            st.subheader("🎯 Recommended Products")
            st.dataframe(recs)

    except Exception as e:
        logger.error(f"Error: {e}")
        st.exception(e)
        st.stop()

# ------------------ FOOTER ------------------
st.caption("""
UI built with Streamlit — backend logic adapted from Flask app.  
Supports: collaborative, content-based, hybrid, and multi-modal recommendation.  
If multi-modal dependencies (torch, torchvision, etc.) are missing, simpler models still run normally.
""")