import os
import math
import joblib
import pandas as pd
import numpy as np
from rapidfuzz import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import streamlit as st

st.set_page_config(page_title="Friendly Food Recommender 🍲✨", page_icon="🍲", layout="centered")

FOOD_CSV = "foods.csv"
RATINGS_CSV = "ratings.csv"
CACHE_FILE = "food_recommender_cache.joblib"
ALPHA = 0.5  # Hybrid balance

# ---------------------------
# Session state defaults
# ---------------------------
def _init_state():
    st.session_state.setdefault("last_recs", [])
    st.session_state.setdefault("last_seed_name", "")
    st.session_state.setdefault("last_method_name", "")
    st.session_state.setdefault("last_show_veg", True)
    st.session_state.setdefault("fb_prompt_for", None)   # validated_name currently prompting fallback
    st.session_state.setdefault("fb_choice", None)       # chosen fallback dish
    st.session_state.setdefault("fb_accept", False)      # user accepted fallback
    st.session_state.setdefault("eval_result", None)     # (rmse, precision, recall, f1)
    st.session_state.setdefault("feedback_submitted", {})# {(seed, method): True}
_init_state()

# ---------------------------
# Data loading
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data_or_cache():
    if os.path.exists(CACHE_FILE):
        cache = joblib.load(CACHE_FILE)
        food_df = cache.get('food_df')
        ratings_df = cache.get('ratings_df')
        if food_df is not None and ratings_df is not None:
            food_df = food_df.copy()
            ratings_df = ratings_df.copy()
            food_df.columns = [c.strip() for c in food_df.columns]
            ratings_df.columns = [c.strip() for c in ratings_df.columns]
            food_df["Name_clean"] = food_df["Name"].astype(str).str.strip().str.lower()
            food_df["C_Type_clean"] = food_df["C_Type"].astype(str).str.strip().str.lower()
            food_df["Veg_Non_clean"] = food_df["Veg_Non"].astype(str).str.strip().str.lower()
            food_df["Describe_clean"] = food_df["Describe"].fillna("").astype(str).str.strip().str.lower()
            if "Avg_Rating" not in food_df.columns or "Rating_Count" not in food_df.columns:
                avg_ratings = (
                    ratings_df.groupby("Food_ID")["Rating"]
                        .agg(['mean','count'])
                        .reset_index()
                        .rename(columns={"mean":"Avg_Rating","count":"Rating_Count"})
                )
                food_df = food_df.merge(avg_ratings, on="Food_ID", how="left")
            food_df["Avg_Rating"] = food_df["Avg_Rating"].fillna(0.0)
            food_df["Rating_Count"] = food_df["Rating_Count"].fillna(0).astype(int)
            return food_df, ratings_df

    if os.path.exists(FOOD_CSV) and os.path.exists(RATINGS_CSV):
        food_df = pd.read_csv(FOOD_CSV)
        ratings_df = pd.read_csv(RATINGS_CSV)

        food_df.columns = [c.strip() for c in food_df.columns]
        ratings_df.columns = [c.strip() for c in ratings_df.columns]

        food_df["Name_clean"] = food_df["Name"].astype(str).str.strip().str.lower()
        food_df["C_Type_clean"] = food_df["C_Type"].astype(str).str.strip().str.lower()
        food_df["Veg_Non_clean"] = food_df["Veg_Non"].astype(str).str.strip().str.lower()
        food_df["Describe_clean"] = food_df["Describe"].fillna("").astype(str).str.strip().str.lower()

        avg_ratings = (
            ratings_df.groupby("Food_ID")["Rating"]
                .agg(['mean','count'])
                .reset_index()
                .rename(columns={"mean":"Avg_Rating","count":"Rating_Count"})
        )
        food_df = food_df.merge(avg_ratings, on="Food_ID", how="left")
        food_df["Avg_Rating"] = food_df["Avg_Rating"].fillna(0.0)
        food_df["Rating_Count"] = food_df["Rating_Count"].fillna(0).astype(int)
        return food_df, ratings_df

    return None, None

food_df, ratings_df = load_data_or_cache()
if food_df is None or ratings_df is None:
    st.error("Missing data. Please place 'food_recommender_cache.joblib' (with food_df & ratings_df) or provide 'foods.csv' and 'ratings.csv' in the app directory.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_vector_space(food_df, ratings_df):
    if os.path.exists(CACHE_FILE):
        cache = joblib.load(CACHE_FILE)
        tfidf_matrix = cache.get('tfidf_matrix')
        cosine_sim_matrix = cache.get('cosine_sim_matrix')
        tfidf = cache.get('tfidf_vectorizer')
        if tfidf is not None and tfidf_matrix is not None and cosine_sim_matrix is not None:
            return tfidf, tfidf_matrix, cosine_sim_matrix

    food_df_local = food_df.copy()
    food_df_local["Content_Text"] = (
        food_df_local["Describe_clean"] + " " +
        food_df_local["C_Type_clean"] + " " +
        food_df_local["Veg_Non_clean"]
    )
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(food_df_local["Content_Text"])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    joblib.dump({
        'food_df': food_df_local,
        'ratings_df': ratings_df,
        'tfidf_matrix': tfidf_matrix,
        'cosine_sim_matrix': cosine_sim_matrix,
        'tfidf_vectorizer': tfidf
    }, CACHE_FILE)

    return tfidf, tfidf_matrix, cosine_sim_matrix

tfidf, tfidf_matrix, cosine_sim_matrix = get_vector_space(food_df, ratings_df)

# ---------------------------
# Helpers
# ---------------------------
def fuzzy_match_choice(query, choices, score_cutoff=50):
    if not choices:
        return None
    res = process.extractOne(query, choices)
    if not res:
        return None
    match, score = res[0], res[1]
    return match if score >= score_cutoff else None

def get_similar_foods_idx(food_idx, top_n=5):
    sims = list(enumerate(cosine_sim_matrix[food_idx]))
    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    res = [(idx, score) for idx, score in sims_sorted if idx != food_idx][:top_n]
    return res

def validate_food_exists(food_name):
    food_name_lower = str(food_name).strip().lower()
    if food_name_lower in food_df["Name_clean"].values:
        original_name = food_df[food_df["Name_clean"]==food_name_lower]["Name"].iloc[0]
        return True, original_name, None
    if len(food_name_lower) >= 3:
        matches = process.extract(food_name_lower, food_df["Name_clean"].tolist(), limit=3)
        suggestions = [food_df[food_df["Name_clean"]==m[0]]["Name"].iloc[0] for m in matches if m[1]>=60]
        if suggestions:
            return False, None, suggestions
    return False, None, None

def format_context_output(recs, seed_name, top_n=5):
    if not recs:
        return []
    seed_food = food_df[food_df["Name"]==seed_name].iloc[0]
    seed_c_type, seed_veg_non = seed_food["C_Type"], seed_food["Veg_Non"]

    formatted = []
    for i, rec in enumerate(recs[:top_n], 1):
        if "Similarity" in rec and isinstance(rec["Similarity"], (int, float)):
            similarity_pct = int(rec["Similarity"] * 100)
        else:
            similarity_pct = int(rec.get("similarity", 0))
        c_type_match = "✅ Same cuisine" if rec["C_Type"]==seed_c_type else "❌ Different cuisine"

        if str(seed_veg_non).lower() == "any":
            veg_match = "🍴 Any preference accepted"
        else:
            if str(rec["Veg_Non"]).lower() == str(seed_veg_non).lower():
                veg_match = "🥗 Veg-match" if str(seed_veg_non).lower().startswith("veg") else "🍗 Non-Veg match"
            else:
                veg_match = "⚠️ Different preference"

        formatted.append({
            "rank": i,
            "name": rec["Name"] if "Name" in rec else rec.get("name",""),
            "similarity": similarity_pct,
            "c_type": rec["C_Type"],
            "c_type_match": c_type_match,
            "veg_non": rec["Veg_Non"],
            "veg_match": veg_match,
            "avg_rating": rec.get("Avg_Rating","N/A"),
            "rating_count": rec.get("Rating_Count","N/A"),
            "source": rec.get("source","N/A")
        })
    return formatted

def display_recommendations(recs, seed_name, method_name, show_veg=True):
    if not recs:
        st.info("No recommendations available.")
        return
    st.subheader(f"{method_name} for '{seed_name}'")
    for rec in recs:
        sim_text = f"({rec.get('similarity',0)}% similar)" if rec.get("similarity") else ""
        st.markdown(f"- **{rec['rank']}. {rec['name']}** {sim_text}")
        if show_veg:
            st.caption(f"Cuisine: {rec['c_type']} | {rec['veg_match']} | ⭐ {rec['avg_rating']} ({rec['rating_count']} reviews)")
        else:
            st.caption(f"Cuisine: {rec['c_type']} | ⭐ {rec['avg_rating']} ({rec['rating_count']} reviews)")

# ---------------------------
# Content-based
# ---------------------------
def content_based(seed_food_name, top_n=5):
    exists, validated_name, _ = validate_food_exists(seed_food_name)
    if not exists:
        return [], None
    match = fuzzy_match_choice(validated_name.lower(), food_df["Name_clean"].tolist(), 40)
    if match is None:
        return [], None
    idx = food_df[food_df["Name_clean"]==match].index[0]
    sim_list = get_similar_foods_idx(idx, top_n)
    recs = []
    for i, sim in sim_list:
        r = food_df.iloc[i]
        recs.append({
            "Name": r["Name"], "C_Type": r["C_Type"], "Veg_Non": r["Veg_Non"],
            "Avg_Rating": r.get("Avg_Rating",0), "Rating_Count": r.get("Rating_Count",0),
            "Similarity": round(sim,3), "source": "Content-Based"
        })
    seed_name = food_df.iloc[idx]["Name"]
    formatted = format_context_output(recs, seed_name, top_n)
    return formatted, seed_name

# ---------------------------
# Collaborative (robust)
# ---------------------------
def collaborative_for_food_robust(seed_food_name, top_n=20):
    exists, validated_name, _ = validate_food_exists(seed_food_name)
    if not exists:
        return [], None
    seed_food_id = food_df.loc[food_df["Name"]==validated_name, "Food_ID"].values[0]
    users_who_liked = ratings_df[ratings_df["Food_ID"]==seed_food_id]["User_ID"].unique()

    if len(users_who_liked) == 0:
        top_foods = ratings_df[ratings_df["Food_ID"]!=seed_food_id].groupby("Food_ID")["Rating"].mean().reset_index()
        top_foods = top_foods.merge(food_df, on="Food_ID").sort_values("Rating", ascending=False).head(top_n)
        recs = [{"Name":r["Name"],"C_Type":r["C_Type"],"Veg_Non":r["Veg_Non"],
                 "Avg_Rating":r["Rating"],
                 "Rating_Count": int(food_df.loc[food_df["Food_ID"]==r["Food_ID"], "Rating_Count"].values[0]) if not food_df.loc[food_df["Food_ID"]==r["Food_ID"], "Rating_Count"].empty else 0,
                 "source":"Collaborative"} for _, r in top_foods.iterrows()]
        return recs, validated_name

    other_ratings = ratings_df[(ratings_df["User_ID"].isin(users_who_liked)) & (ratings_df["Food_ID"]!=seed_food_id)]
    if other_ratings.empty:
        top_foods = ratings_df[ratings_df["Food_ID"]!=seed_food_id].groupby("Food_ID")["Rating"].mean().reset_index()
        top_foods = top_foods.merge(food_df, on="Food_ID").sort_values("Rating", ascending=False).head(top_n)
        recs = [{"Name":r["Name"],"C_Type":r["C_Type"],"Veg_Non":r["Veg_Non"],
                 "Avg_Rating":r["Rating"],
                 "Rating_Count": int(food_df.loc[food_df["Food_ID"]==r["Food_ID"], "Rating_Count"].values[0]) if not food_df.loc[food_df["Food_ID"]==r["Food_ID"], "Rating_Count"].empty else 0,
                 "source":"Collaborative"} for _, r in top_foods.iterrows()]
        return recs, validated_name

    agg = other_ratings.groupby("Food_ID")["Rating"].agg(['mean','count']).reset_index()
    agg = agg.merge(food_df, on="Food_ID").sort_values(by=['count','mean'], ascending=False).head(top_n)
    recs = [{"Name":r["Name"],"C_Type":r["C_Type"],"Veg_Non":r["Veg_Non"],
             "Avg_Rating":float(r['mean']), "Rating_Count":int(r['count']), "source":"Collaborative"} for _, r in agg.iterrows()]
    return recs, validated_name

def format_collab_recommendations(recs, seed_name):
    formatted = []
    seed_veg = food_df[food_df["Name"]==seed_name]["Veg_Non"].values[0]
    for i, r in enumerate(recs, 1):
        if str(seed_veg).lower() == "any":
            veg_match = "🍴 Any preference accepted"
        else:
            if str(r["Veg_Non"]).lower() == str(seed_veg).lower():
                veg_match = "🥗 Veg-match" if str(seed_veg).lower().startswith("veg") else "🍗 Non-Veg match"
            else:
                veg_match = "⚠️ Different preference"
        formatted.append({
            "rank": i,
            "name": r["Name"],
            "similarity": 0,
            "c_type": r["C_Type"],
            "veg_non": r["Veg_Non"],
            "veg_match": veg_match,
            "avg_rating": r.get("Avg_Rating",0),
            "rating_count": r.get("Rating_Count",0),
            "source": r.get("source","Collaborative")
        })
    return formatted

# ---------------------------
# Collaborative with fallback (stateful; buttons handled at top-level)
# ---------------------------
def collaborative_with_fallback_ui_with_message(seed_food_name, top_n=5):
    exists, validated_name, _ = validate_food_exists(seed_food_name)
    if not exists:
        st.warning(f"'{seed_food_name}' not found in database.")
        return

    seed_food_id = food_df.loc[food_df["Name"]==validated_name, "Food_ID"].values[0]
    users_who_liked = ratings_df[ratings_df["Food_ID"]==seed_food_id]["User_ID"].unique()

    if len(users_who_liked) > 0:
        recs, seed_name = collaborative_for_food_robust(validated_name, top_n)
        recs = format_collab_recommendations(recs, seed_name)
        st.session_state.last_recs = recs
        st.session_state.last_seed_name = seed_name
        st.session_state.last_method_name = "👥 Collaborative Filtering"
        st.session_state.fb_prompt_for = None
        st.session_state.fb_choice = None
        st.session_state.fb_accept = False
        return

    idx = food_df[food_df["Name"]==validated_name].index[0]
    sims = get_similar_foods_idx(idx, top_n=5)
    suggestions = [food_df.iloc[i]["Name"] for i,_ in sims if food_df.iloc[i]["Rating_Count"] > 0]

    if not suggestions:
        st.info("😔 Sorry, no similar dishes with ratings found in the dataset.")
        st.session_state.fb_prompt_for = None
        st.session_state.fb_choice = None
        st.session_state.fb_accept = False
        return

    st.session_state.fb_prompt_for = validated_name
    st.session_state.fb_choice = suggestions[0]
    st.session_state.fb_accept = False

# ---------------------------
# Hybrid
# ---------------------------
def hybrid_balanced(seed_food_name, top_n=5):
    content_rec, seed_name = content_based(seed_food_name, top_n=20)
    if not seed_name:
        return [], None
    collab_rec, _ = collaborative_for_food_robust(seed_food_name, top_n=20)
    collab_rec = format_collab_recommendations(collab_rec, seed_name)

    combined = {}
    max_similarity = max([r.get('similarity', 0) for r in content_rec], default=1)
    max_collab_rating = max([r.get('avg_rating', 0) for r in collab_rec], default=5)

    for r in content_rec:
        sim_score = (r.get('similarity', 0) / max_similarity) if max_similarity > 0 else 0
        combined[r['name']] = {
            "Name": r['name'],
            "C_Type": r['c_type'],
            "Veg_Non": r['veg_non'],
            "Similarity_norm": sim_score,
            "Collab_norm": 0,
            "Avg_Rating": r['avg_rating'],
            "Rating_Count": r['rating_count'],
            "source": "Hybrid"
        }

    for r in collab_rec:
        collab_score = r.get('avg_rating', 0) / max_collab_rating if max_collab_rating > 0 else 0
        if r['name'] in combined:
            combined[r['name']]["Collab_norm"] = collab_score
        else:
            combined[r['name']] = {
                "Name": r['name'],
                "C_Type": r['c_type'],
                "Veg_Non": r['veg_non'],
                "Similarity_norm": 0,
                "Collab_norm": collab_score,
                "Avg_Rating": r['avg_rating'],
                "Rating_Count": r['rating_count'],
                "source": "Hybrid"
            }

    for r in combined.values():
        r['final_score'] = ALPHA * r['Similarity_norm'] + (1 - ALPHA) * r['Collab_norm']

    final_list = sorted(combined.values(), key=lambda x: x['final_score'], reverse=True)
    formatted = format_context_output(final_list, seed_name, top_n)
    return formatted, seed_name

# ---------------------------
# Evaluation
# ---------------------------
def compute_rmse_actual(recs, seed_food_name):
    actual_ratings = []
    predicted_ratings = []
    for r in recs:
        rn = r.get('name', '')
        if not rn:
            continue
        food_rows = food_df[food_df["Name"]==rn]
        if food_rows.empty:
            continue
        food_id = food_rows["Food_ID"].values[0]
        user_ratings = ratings_df[ratings_df["Food_ID"]==food_id]["Rating"].tolist()
        if user_ratings:
            actual_ratings.extend(user_ratings)
            predicted_ratings.extend([r.get('avg_rating', 0)] * len(user_ratings))
    if not actual_ratings:
        return 0.0
    mse = mean_squared_error(actual_ratings, predicted_ratings)
    return math.sqrt(mse)

def compute_precision_recall_f1_actual(recs):
    recommended = [r.get('name','') for r in recs if r.get('name')]
    relevant = food_df[food_df['Avg_Rating'] >= 4.0]['Name'].tolist()
    recommended_set = set(recommended)
    relevant_set = set(relevant)
    tp = len(recommended_set & relevant_set)
    precision = tp / len(recommended_set) if recommended_set else 0
    recall = tp / len(relevant_set) if relevant_set else 0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
    return precision, recall, f1

def display_evaluation_results(result):
    if not result:
        st.info("No evaluation available.")
        return
    rmse_score, precision, recall, f1 = result
    st.metric("RMSE (Predicted vs Actual)", f"{rmse_score:.2f}")
    colp, colr, colf = st.columns(3)
    colp.metric("Precision", f"{precision:.2f}")
    colr.metric("Recall", f"{recall:.2f}")
    colf.metric("F1 Score", f"{f1:.2f}")

# ---------------------------
# Satisfaction
# ---------------------------
def ask_system_satisfaction(seed_name, method_name):
    key_prefix = f"{seed_name}_{method_name}"
    submitted = st.session_state.feedback_submitted.get(key_prefix, False)
    score = st.slider("Your rating:", 1, 5, 4, key=f"slider_{key_prefix}")
    if not submitted:
        if st.button("✅ Submit Feedback", key=f"submit_{key_prefix}"):
            st.session_state.feedback_submitted[key_prefix] = True
            submitted = True
    if submitted:
        emoji = {1:"😞", 2:"😕", 3:"😐", 4:"🙂", 5:"🤩"}
        st.success(f"You rated the system: {score}/5 {emoji.get(score,'')}. Thanks for your feedback!")

# ---------------------------
# UI
# ---------------------------
st.title("Friendly Food Recommender 🍲✨")
st.write("Let’s find you something delicious today 😋")

cuisine_options = ["Any"] + sorted([c for c in food_df["C_Type_clean"].dropna().unique().tolist() if c])
veg_options = ["Any"] + sorted([v for v in food_df["Veg_Non_clean"].dropna().unique().tolist() if v])

colA, colB = st.columns(2)
with colA:
    cuisine_choice = st.selectbox("🍛 Cuisine:", cuisine_options, index=0)
with colB:
    veg_choice = st.selectbox("🥦 Preference:", veg_options, index=0)

filtered_df = food_df.copy()
if cuisine_choice != "Any":
    filtered_df = filtered_df[filtered_df["C_Type_clean"] == cuisine_choice]
if veg_choice != "Any":
    filtered_df = filtered_df[filtered_df["Veg_Non_clean"] == veg_choice]
clean_food_df = filtered_df.dropna(subset=["Name"]).drop_duplicates(subset=["Name_clean"])
dish_list = sorted(clean_food_df["Name"].tolist()) if not clean_food_df.empty else sorted(food_df["Name"].dropna().unique().tolist())
dish = st.selectbox("🍲 Dish:", dish_list if dish_list else food_df["Name"].tolist())

method = st.selectbox(
    "⚡ Style:",
    options=[
        "🔍 Similar Foods (Content-Based)",
        "👥 What Others Liked (Collaborative)",
        "🤝 Best of Both (Hybrid)",
    ],
)

col1, col2, col3 = st.columns(3)
with col1:
    show_btn = st.button("✨ Show My Recommendations", key="show_btn")
with col2:
    eval_btn = st.button("📊 Evaluate System", key="eval_btn")
with col3:
    clear_btn = st.button("🧹 Clear", key="clear_btn")

# Clear
if clear_btn:
    st.session_state.last_recs = []
    st.session_state.last_seed_name = ""
    st.session_state.last_method_name = ""
    st.session_state.last_show_veg = True
    st.session_state.fb_prompt_for = None
    st.session_state.fb_choice = None
    st.session_state.fb_accept = False
    st.session_state.eval_result = None

# Show recommendations
if show_btn:
    if not dish or dish not in food_df["Name"].values:
        st.warning("Please choose a valid dish.")
    else:
        if method.startswith("🔍"):
            recs, seed_name = content_based(dish, top_n=5)
            st.session_state.last_recs = recs
            st.session_state.last_seed_name = seed_name
            st.session_state.last_method_name = "🔍 Content-Based"
            st.session_state.last_show_veg = (veg_choice != "Any")
            st.session_state.fb_prompt_for = None
            st.session_state.eval_result = None
        elif method.startswith("👥"):
            st.session_state.last_show_veg = (veg_choice != "Any")
            st.session_state.eval_result = None
            collaborative_with_fallback_ui_with_message(dish, top_n=5)
        else:
            recs, seed_name = hybrid_balanced(dish, top_n=5)
            st.session_state.last_recs = recs
            st.session_state.last_seed_name = seed_name
            st.session_state.last_method_name = "🤝 Hybrid Mix"
            st.session_state.last_show_veg = (veg_choice != "Any")
            st.session_state.fb_prompt_for = None
            st.session_state.eval_result = None

# Render fallback prompt and handle Yes/No on every rerun
if st.session_state.fb_prompt_for and not st.session_state.fb_accept:
    name = st.session_state.fb_prompt_for
    fallback_choice = st.session_state.fb_choice
    st.warning(f"'{name}' has no ratings. Try a similar dish: {fallback_choice}?")
    colY, colN = st.columns(2)
    yes = colY.button(f"Yes, use {fallback_choice}", key=f"use_fb_{name}")
    no = colN.button("No, skip", key=f"skip_fb_{name}")

    if yes:
        recs, seed_name = collaborative_for_food_robust(fallback_choice, top_n=5)
        recs = format_collab_recommendations(recs, seed_name)
        st.session_state.last_recs = recs
        st.session_state.last_seed_name = seed_name
        st.session_state.last_method_name = "👥 Collaborative Filtering (Fallback)"
        st.session_state.fb_accept = True
        st.rerun()
    elif no:
        st.info("No collaborative recommendations will be generated.")
        st.session_state.fb_prompt_for = None
        st.session_state.fb_choice = None
        st.session_state.fb_accept = False

# Always render last recommendations (if present)
if st.session_state.last_recs and st.session_state.last_seed_name:
    display_recommendations(
        st.session_state.last_recs,
        st.session_state.last_seed_name,
        st.session_state.last_method_name,
        show_veg=st.session_state.get("last_show_veg", True)
    )
    ask_system_satisfaction(st.session_state.last_seed_name, st.session_state.last_method_name)

# Evaluate on demand
if eval_btn:
    if not dish or dish not in food_df["Name"].values:
        st.warning("Please pick a valid dish from the list 📝")
    else:
        if method.startswith("🔍"):
            recs, seed_name = content_based(dish, top_n=5)
        elif method.startswith("👥"):
            raw, seed_name = collaborative_for_food_robust(dish, top_n=5)
            recs = format_collab_recommendations(raw, seed_name) if seed_name else []
        else:
            recs, seed_name = hybrid_balanced(dish, top_n=5)

        if not recs or not seed_name:
            st.info("No recommendations to evaluate.")
            st.session_state.eval_result = None
        else:
            rmse_score = compute_rmse_actual(recs, seed_name)
            precision, recall, f1 = compute_precision_recall_f1_actual(recs)
            st.session_state.eval_result = (rmse_score, precision, recall, f1)

# Render evaluation (if available)
if st.session_state.eval_result:
    st.subheader("Evaluation")
    display_evaluation_results(st.session_state.eval_result)

st.caption("🚪 Close the browser tab to exit the app.")