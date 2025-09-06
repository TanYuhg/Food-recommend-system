# ================================
# FRIENDLY FOOD RECOMMENDER SYSTEM 🍲✨
# Streamlit Version - Fully Matches Colab
# ================================
import streamlit as st
import os
import joblib
import math
from rapidfuzz import process
from sklearn.metrics import mean_squared_error

# ================================
# Load Cache
# ================================
CACHE_FILE = "food_recommender_cache.joblib"

if not os.path.exists(CACHE_FILE):
    st.error("Cache file not found! Make sure 'food_recommender_cache.joblib' is in the repo.")
    st.stop()

cache = joblib.load(CACHE_FILE)
food_df = cache['food_df']
ratings_df = cache['ratings_df']
tfidf_matrix = cache['tfidf_matrix']
cosine_sim_matrix = cache['cosine_sim_matrix']
tfidf = cache['tfidf_vectorizer']

# ================================
# Helper Functions
# ================================
def fuzzy_match_choice(query, choices, score_cutoff=50):
    if not choices: return None
    res = process.extractOne(query, choices)
    if not res: return None
    match, score = res[0], res[1]
    return match if score >= score_cutoff else None

def get_similar_foods_idx(food_idx, top_n=5):
    sims = list(enumerate(cosine_sim_matrix[food_idx]))
    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    return [(idx, score) for idx, score in sims_sorted if idx != food_idx][:top_n]

def validate_food_exists(food_name):
    food_name_lower = food_name.strip().lower()
    if food_name_lower in food_df["Name_clean"].values:
        original_name = food_df[food_df["Name_clean"]==food_name_lower]["Name"].iloc[0]
        return True, original_name, None
    if len(food_name_lower) >= 3:
        matches = process.extract(food_name_lower, food_df["Name_clean"].tolist(), limit=3)
        suggestions = [food_df[food_df["Name_clean"]==m[0]]["Name"].iloc[0] for m in matches if m[1]>=60]
        if suggestions: return False, None, suggestions
    return False, None, None

def format_context_output(recs, seed_name, top_n=5):
    if not recs: return []
    seed_food = food_df[food_df["Name"]==seed_name].iloc[0]
    seed_c_type, seed_veg_non = seed_food["C_Type"], seed_food["Veg_Non"]
    formatted = []
    for i, rec in enumerate(recs[:top_n],1):
        similarity_pct = int(rec.get("Similarity",0)*100)
        c_type_match = "✅ Same cuisine" if rec["C_Type"]==seed_c_type else "❌ Different cuisine"
        # Veg/Non-Veg + Any
        if seed_veg_non.lower() == "any":
            veg_match = "🍴 Any preference accepted"
        else:
            if rec["Veg_Non"].lower() == seed_veg_non.lower():
                veg_match = "🥗 Veg-match" if seed_veg_non.lower().startswith("veg") else "🍗 Non-Veg match"
            else:
                veg_match = "⚠️ Different preference"
        formatted.append({
            "rank": i,
            "name": rec["Name"],
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

# ================================
# Recommendation Functions
# ================================
def content_based(seed_food_name, top_n=5):
    exists, validated_name, _ = validate_food_exists(seed_food_name)
    if not exists: return [], None
    match = fuzzy_match_choice(validated_name.lower(), food_df["Name_clean"].tolist(), 40)
    if match is None: return [], None
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
    return format_context_output(recs, seed_name, top_n), seed_name

def collaborative_for_food_robust(seed_food_name, top_n=20):
    exists, validated_name, _ = validate_food_exists(seed_food_name)
    if not exists: return [], None
    seed_food_id = food_df.loc[food_df["Name"]==validated_name, "Food_ID"].values[0]
    users_who_liked = ratings_df[ratings_df["Food_ID"]==seed_food_id]["User_ID"].unique()
    other_ratings = ratings_df[(ratings_df["User_ID"].isin(users_who_liked)) & (ratings_df["Food_ID"]!=seed_food_id)]
    if other_ratings.empty:
        top_foods = ratings_df[ratings_df["Food_ID"]!=seed_food_id].groupby("Food_ID")["Rating"].mean().reset_index()
        top_foods = top_foods.merge(food_df, on="Food_ID").sort_values("Rating", ascending=False).head(top_n)
        recs = [{"Name":r["Name"],"C_Type":r["C_Type"],"Veg_Non":r["Veg_Non"],
                 "Avg_Rating":r["Rating"],"Rating_Count": food_df.loc[food_df["Food_ID"]==r["Food_ID"], "Rating_Count"].values[0],
                 "source":"Collaborative"} for _, r in top_foods.iterrows()]
        return recs, validated_name
    agg = other_ratings.groupby("Food_ID")["Rating"].agg(['mean','count']).reset_index()
    agg = agg.merge(food_df, on="Food_ID").sort_values(by=['count','mean'], ascending=False).head(top_n)
    recs = [{"Name":r["Name"],"C_Type":r["C_Type"],"Veg_Non":r["Veg_Non"],
             "Avg_Rating": r['mean'], "Rating_Count":int(r['count']), "source":"Collaborative"} for _, r in agg.iterrows()]
    return recs, validated_name

def format_collab_recommendations(recs, seed_name):
    formatted = []
    seed_veg = food_df[food_df["Name"]==seed_name]["Veg_Non"].values[0]
    for i, r in enumerate(recs, 1):
        if seed_veg.lower() == "any":
            veg_match = "🍴 Any preference accepted"
        else:
            veg_match = "🥗 Veg-match" if r["Veg_Non"]==seed_veg else "🍗 Non-Veg match"
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

ALPHA = 0.5
def hybrid_balanced(seed_food_name, top_n=5):
    content_rec, seed_name = content_based(seed_food_name, top_n=20)
    collab_rec, _ = collaborative_for_food_robust(seed_food_name, top_n=20)
    collab_rec = format_collab_recommendations(collab_rec, seed_name)
    combined = {}
    max_similarity = max([r['similarity'] for r in content_rec], default=1)
    max_collab_rating = max([r['avg_rating'] for r in collab_rec], default=5)
    for r in content_rec:
        sim_score = r['similarity'] / max_similarity if max_similarity>0 else 0
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
        collab_score = r['avg_rating'] / max_collab_rating if max_collab_rating>0 else 0
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
        r['final_score'] = ALPHA * r['Similarity_norm'] + (1-ALPHA) * r['Collab_norm']
    final_list = sorted(combined.values(), key=lambda x: x['final_score'], reverse=True)
    return format_context_output(final_list, seed_name, top_n), seed_name

# ================================
# Evaluation Metrics
# ================================
def compute_rmse_actual(recs, seed_food_name):
    actual_ratings = []
    predicted_ratings = []
    for r in recs:
        r = {k.lower():v for k,v in r.items()}
        food_id = food_df[food_df["Name"]==r['name']]["Food_ID"].values[0]
        user_ratings = ratings_df[ratings_df["Food_ID"]==food_id]["Rating"].tolist()
        if user_ratings:
            actual_ratings.extend(user_ratings)
            predicted_ratings.extend([r['avg_rating']]*len(user_ratings))
    if not actual_ratings: return 0
    return math.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

def compute_precision_recall_f1_actual(recs):
    recommended = [r['name'] for r in recs]
    relevant = food_df[food_df['Avg_Rating']>=4.0]['Name'].tolist()
    recommended_set, relevant_set = set(recommended), set(relevant)
    tp = len(recommended_set & relevant_set)
    precision = tp / len(recommended_set) if recommended_set else 0
    recall = tp / len(relevant_set) if relevant_set else 0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0
    return precision, recall, f1

def display_evaluation_results(recs, seed_name):
    st.subheader("📊 Evaluation Metrics")
    rmse_score = compute_rmse_actual(recs, seed_name)
    st.markdown(f"**RMSE (Predicted vs Actual ratings):** {rmse_score:.2f}")
    precision, recall, f1 = compute_precision_recall_f1_actual(recs)
    st.markdown(f"**Precision:** {precision:.2f} | **Recall:** {recall:.2f} | **F1 Score:** {f1:.2f}")

# ================================
# Fuzzy Dish Suggestions (Fixed)
# ================================
def get_food_suggestions(query, cuisine_choice="Any", veg_choice="Any", max_suggestions=20, score_cutoff=50):
    df = food_df.copy()
    # Apply cuisine/veg filters first on cleaned columns
    if cuisine_choice != "Any":
        df = df[df["C_Type_clean"]==cuisine_choice.lower()]
    if veg_choice != "Any":
        df = df[df["Veg_Non_clean"]==veg_choice.lower()]
    if not query or len(query.strip())<2:
        options = sorted(df["Name"].tolist())
    else:
        query = query.lower().strip()
        matches = process.extract(query, df["Name_clean"].tolist(), limit=max_suggestions)
        options = [df[df["Name_clean"]==m[0]]["Name"].iloc[0] for m in matches if m[1]>=score_cutoff]
        if not options:
            options = sorted(df["Name"].tolist())
    return options

# ================================
# Streamlit UI
# ================================
st.title("🍲 Friendly Food Recommender System 🍴✨")
st.sidebar.header("Filter & Options")

# Filters
cuisine_options = ["Any"] + sorted(food_df["C_Type_clean"].dropna().unique())
veg_options = ["Any"] + sorted(food_df["Veg_Non_clean"].dropna().unique())

cuisine_choice = st.sidebar.selectbox("🍛 Cuisine:", cuisine_options)
veg_choice = st.sidebar.selectbox("🥦 Preference:", veg_options)
method_choice = st.sidebar.radio("⚡ Recommendation Style:", 
                                 ["🔍 Content-Based","👥 Collaborative","🤝 Hybrid"])

# Dish input with fuzzy search
food_query = st.text_input("🍲 Type to search/select a dish:", "")
food_options = get_food_suggestions(food_query, cuisine_choice, veg_choice)

food_name = st.selectbox("🍲 Select a Dish:", food_options)

# ================================
# Session state for feedback & results
# ================================
if "results" not in st.session_state:
    st.session_state.results = {}   # store recs per method
if "last_method" not in st.session_state:
    st.session_state.last_method = None
if "feedback_score" not in st.session_state:
    st.session_state.feedback_score = 4
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "last_seed" not in st.session_state:
    st.session_state.last_seed = None

# ================================
# Recommendations Button
# ================================
if st.button("✨ Show Recommendations"):
    if not food_name:
        st.warning("⚠️ Please select a valid dish!")
    else:
        st.session_state.feedback_submitted = False
        if method_choice == "🔍 Content-Based":
            recs, seed_name = content_based(food_name, top_n=5)
        elif method_choice == "👥 Collaborative":
            recs, seed_name = collaborative_for_food_robust(food_name, top_n=5)
            recs = format_collab_recommendations(recs, seed_name)
        else:
            recs, seed_name = hybrid_balanced(food_name, top_n=5)

        # save results tied to method
        st.session_state.results[method_choice] = {
            "recs": recs,
            "seed": seed_name
        }
        st.session_state.last_method = method_choice

# ================================
# Display Recommendations
# ================================

if st.session_state.last_method in st.session_state.results:
    data = st.session_state.results[st.session_state.last_method]
    recs, seed_name = data["recs"], data["seed"]

    st.subheader(f"🍽 Recommendations for *{seed_name}* ({st.session_state.last_method})")
    for r in recs:
        sim_text = "👍 Loved by others"
        if st.session_state.last_method == "🔍 Content-Based":
            sim_text = f"✨ {r.get('similarity',0)}% similar"
        elif st.session_state.last_method == "🤝 Hybrid":
            sim_text = "⚖️ Balanced pick (flavor + ratings)"
        emoji = "🥦" if str(r["veg_non"]).lower().startswith("veg") else "🍗"
        st.markdown(f"**{r['rank']}. {r['name']} {emoji}**  \nCuisine: {r['c_type']} | {r['veg_match']} | {sim_text}  \n⭐ {r['avg_rating']:.1f} ({r['rating_count']} reviews)")
    # Feedback Section
    st.subheader("📋 How satisfied are you with these recommendations?")
    st.session_state.feedback_score = st.slider(
        "Your rating (1 = Low, 5 = Excellent):",
        1, 5, st.session_state.feedback_score
    )
    if st.button("✅ Submit Feedback"):
        st.session_state.feedback_submitted = True

    if st.session_state.feedback_submitted:
        emoji_feedback = {1:"😞",2:"😕",3:"😐",4:"🙂",5:"🤩"}
        st.success(f"⭐ You rated the system: {st.session_state.feedback_score}/5 {emoji_feedback[st.session_state.feedback_score]}")
        st.info("👍 Thanks for your feedback! This helps improve future recommendations.")

    # Evaluation Metrics
    display_evaluation_results(recs, seed_name)

# ================================
# Exit
# ================================
if st.button("🚪 Exit"):
    st.success("👋 Thanks for using the Friendly Food Recommender! See you next time! 🍴✨")
    st.stop()
