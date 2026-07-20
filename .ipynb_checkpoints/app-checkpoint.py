# This file is the scource code for deploying the recommendation algorithm constructed in recommender_ipynb as a Web App to Streamlit.
# From our cleaned dataset df_clean, we use the features Website and Header image in the UI of the app to build clickable links in the app. 


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Set up a clean, wide page layout
st.set_page_config(page_title = "Steam Games Recommender", page_icon = "🎮", layout = "wide", initial_sidebar_state = "expanded")

st.markdown(
    """
    <style>
        
        /* 1. Change button color to blue. */
        div.stButton > button[kind="primary"] {background-color: #1078ff !important; color: white !important; border: none !important;}
        

        /* 2. Change the background and text color of selected items in search field. */
        span[data-baseweb="tag"] {background-color: #1078ff !important; color: white !important;}
        

        /* 3. Ensure the "x" icon inside the tags stays readable. */
        span[data-baseweb="tag"] svg {fill: hite !important;}


        /* 4. When 3 games are selected, hide the dropdown. */
        html:has(span[data-baseweb="tag"]:nth-of-type(3)) div[role="listbox"],
        html:has(span[data-baseweb="tag"]:nth-of-type(3)) div[data-baseweb="popover"] {display: none !important; visibility: hidden !important; opacity: 0 
        !important; height: 0px !important;}
    </style>
    """,
    unsafe_allow_html = True)


@st.cache_data
def load_data():
    
    df_clean = pd.read_csv('data/games_clean.csv') 
    df_topten = pd.read_csv('data/df_topten.csv')
    df_free_topten = pd.read_csv('data/df_free_topten.csv')
    df_paid_topten = pd.read_csv('data/df_paid_topten.csv')
    
    # Fill empty characteristics strings.
    df_clean['Characteristics'] = df_clean['Characteristics'].fillna('')
    df_topten['Characteristics'] = df_topten['Characteristics'].fillna('')
    df_paid_topten['Characteristics'] = df_paid_topten['Characteristics'].fillna('')
    df_free_topten['Characteristics'] = df_free_topten['Characteristics'].fillna('')
    
    return df_clean, df_paid_topten, df_free_topten, df_topten


    

# ---------Recommender architecture starts: taken as-it-is from recommender.ipynb-----------   

@st.cache_resource

# This function builds the matrices that the RS will accept onto which it can compute cosine similarity.

def build_matrices(df_clean, df_paid_topten, df_free_topten):
    char_vectorizer = CountVectorizer(token_pattern = r'(?u)\b\w+\b') 
    char_vectorizer.fit(df_topten['Characteristics'].fillna(''))

    # I. Matrix for paid games (Using 50/40/10 weights)

    # Transform (uses the master vocabulary)
    char_sparse_paid = char_vectorizer.transform(df_paid_topten['Characteristics'].fillna(''))
    char_weighted_paid = char_sparse_paid * np.sqrt(0.5)
    
    # Extract and weight votes & revenue.
    votes_weighted_paid = (df_paid_topten['Bayesian votes'].values * np.sqrt(0.4)).reshape(-1, 1) # reshape(-1,1) converts a 1d matrix to 2d so that hstack works correctly.
    revenue_weighted_paid = (df_paid_topten['Estimated yearly revenue'].values * np.sqrt(0.1)).reshape(-1, 1)
    
    # Stack to create the paid master feature matrix.
    X_paid = hstack([char_weighted_paid, votes_weighted_paid, revenue_weighted_paid]).tocsr()
    
    
    # II. Matrix for free games (Using 50/40/10 weights)
    
    
    # Transform (uses the master vocabulary).
    char_sparse_free = char_vectorizer.transform(df_free_topten['Characteristics'].fillna(''))
    char_weighted_free = char_sparse_free * np.sqrt(0.5)
    
    # Extract and weight votes & owners.
    votes_weighted_free = (df_free_topten['Bayesian votes'].values * np.sqrt(0.4)).reshape(-1, 1)
    owners_weighted_free = (df_free_topten['Estimated owners'].values * np.sqrt(0.1)).reshape(-1, 1)
    
    # Stack to create the free master feature matrix.
    X_free = hstack([char_weighted_free, votes_weighted_free, owners_weighted_free]).tocsr()

    return char_vectorizer, X_paid, X_free


# Initialize data and models.
df_clean, df_paid_topten, df_free_topten, df_topten = load_data()
char_vectorizer, X_paid, X_free = build_matrices(df_clean, df_paid_topten, df_free_topten)


# Recommender algorithm: Recommends 8 paid games and 2 free games based on a list of input games.


#  Function Parameters:
#    - input_titles: List of 3 strings (the user's input games)
#    - df_paid_topten: Subset of cleaned & filtered dataset (df_topten) for paid recommendations
#    - X_paid: Weighted feature matrix for df_paid_topten
#    - df_free_topten: Subset of cleaned & filtered dataset (df_topten) for free recommendations
#    - X_free: Weighted feature matrix for df_free_topten
#    - df_master: Cleaned & filtered master dataset (df_topten)
#    - char_vectorizer: Count vectorizer fit on Characteristics


def recommend_games(input_titles, df_paid_topten, X_paid, df_free_topten, X_free, df_master, char_vectorizer):
    
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import pandas as pd
    from scipy.sparse import hstack

    # Standardize input titles for robust matching
    input_titles_clean = [title.strip().lower() for title in input_titles]
    
    # 1. Locate the input games in df_master
    matched_games = df_master[df_master['Name'].str.strip().str.lower().isin(input_titles_clean)]
    
    # Check if we have at least one valid game in df_master.
    
    if matched_games.empty:
        # Fallback 1: None of the input games exist on Steam at all
        message = "We don't have enough information to recommend games based on the games you input but here are some great games you can check out!"
        fallback_paid = df_paid_topten.sort_values(by = 'Bayesian votes', ascending = False).head(8)
        fallback_free = df_free_topten.sort_values(by = 'Bayesian votes', ascending = False).head(2)
        return fallback_paid, fallback_free, message
    

# Check if all input games are missing Characteristics data in our dataset.

    
    # Clean up the strings to check for empty/NaN values
    has_characteristics = matched_games['Characteristics'].fillna('').str.strip() != ""
    
    if not has_characteristics.any():  # If ALL matched games have missing Characteristics
        message = "We don't have enough information to recommend games based on the games you input but here are some great games you can check out!"
        fallback_paid = df_paid_topten.sort_values(by = 'Bayesian votes', ascending=False).head(8)
        fallback_free = df_free_topten.sort_values(by = 'Bayesian votes', ascending=False).head(2)
        return fallback_paid, fallback_free, message

    
    # Construct feature vectors for input games.
    
    # Filter matched games down to the ones that actually have characteristics.
    profile_games = matched_games[has_characteristics]
    
    # Project their characteristics into the vectorizer space and apply characteristics weight (50%)
    char_sparse_query = char_vectorizer.transform(profile_games['Characteristics'].fillna(''))
    char_weighted_query = char_sparse_query * np.sqrt(0.5)
    
    # Extract popularity metrics
    votes_query = profile_games['Bayesian votes'].values.reshape(-1, 1)
    revenue_query = profile_games['Estimated yearly revenue'].values.reshape(-1, 1)
    owners_query = profile_games['Estimated owners'].values.reshape(-1, 1)
    
    # Build the input games' vector for comparing with paid games
    votes_weighted_query_paid = (votes_query * np.sqrt(0.4)).reshape(-1, 1)
    revenue_weighted_query_paid = (revenue_query * np.sqrt(0.1)).reshape(-1, 1)
    
    # Stack the features for the inputs and take the mean to make a single 1D query vector
    query_matrix_paid = hstack([char_weighted_query, votes_weighted_query_paid, revenue_weighted_query_paid]).tocsr()
    query_vector_paid = np.mean(query_matrix_paid.toarray(), axis=0).reshape(1, -1)

    # Build the query vector for comparing with free games.
    votes_weighted_query_free = (votes_query * np.sqrt(0.4)).reshape(-1, 1)
    owners_weighted_query_free = (owners_query * np.sqrt(0.1)).reshape(-1, 1)
    
    query_matrix_free = hstack([char_weighted_query, votes_weighted_query_free, owners_weighted_query_free]).tocsr()
    query_vector_free = np.mean(query_matrix_free.toarray(), axis=0).reshape(1, -1)

# Compute cosine similarity and obtain recommendations.
    
    # Compute similarities and compare with the paid target matrix
    paid_similarities = cosine_similarity(query_vector_paid, X_paid).flatten()
    df_paid_results = df_paid_topten.copy()
    df_paid_results['similarity_score'] = paid_similarities
    
    # Filter out inputs
    df_paid_results = df_paid_results[~df_paid_results['Name'].str.strip().str.lower().isin(input_titles_clean)]
    top_paid = df_paid_results.sort_values(by = 'similarity_score', ascending = False).head(8)

    # Compute similarities and compare against the free target matrix.
    free_similarities = cosine_similarity(query_vector_free, X_free).flatten()
    df_free_results = df_free_topten.copy()
    df_free_results['similarity_score'] = free_similarities
    
    # Filter out inputs
    df_free_results = df_free_results[~df_free_results['Name'].str.strip().str.lower().isin(input_titles_clean)]
    top_free = df_free_results.sort_values(by = 'similarity_score', ascending = False).head(2)

    return top_paid, top_free, None

    

# ---------Recommender architecture ends: taken as-it-is from recommender.ipynb-----------   

    
# Generate html links for recommendations.

def render_game_card(game_row):
    
    name = game_row['Name']
    website = game_row['Website']
    header_image = game_row['Header image']
    
    if pd.isna(header_image) or str(header_image).strip() == "":
        header_image = "https://via.placeholder.com/460x215.png?text=No+Image+Available"
        
    if pd.isna(website) or str(website).strip() == "":
        website = f"https://store.steampowered.com/search/?term={name.replace(' ', '+')}"

    card_html = f"""
    <div style="
        border-radius: 8px; 
        background-color: #1b2838; 
        padding: 10px; 
        margin-bottom: 20px; 
        border: 1px solid #2a475e;
        transition: transform 0.2s;
    ">
        <a href="{website}" target="_blank" style="text-decoration: none; color: white;">
            <img src="{header_image}" style="width: 100%; border-radius: 4px; display: block; margin-bottom: 8px;" />
            <div style="
                font-weight: bold; 
                font-size: 1.1em; 
                color: #66c0f4; 
                line-height: 1.2; 
                min-height: 2.4em; 
                display: -webkit-box; 
                -webkit-line-clamp: 2; 
                -webkit-box-orient: vertical; 
                overflow: hidden;
            ">
                {name}
            </div>
        </a>
    </div>
    """
    return card_html


# Sidebar 

with st.sidebar:
    
    # Centered "About the Engine" heading.
    st.markdown("<h1 style = 'text-align: center;'>About the Engine</h1>", unsafe_allow_html = True)
    
    # Justified app description
    st.markdown("""
    <div style = "text-align: justify;">
    This content-based recommender system maps the underlying <b>thematic feel and mechanics</b> of games using cosine similarity on processed gameplay characteristics and quality metrics. It accepts up to 3 games as <b>input</b> and outputs <b>10</b> high-quality recommendations: 8 premium (paid) and 2 free-to-play. Please click <a href = "https://github.com/shujoshi-git/steam-games-recommender" target = "_blank" style ="color: #66c0f4; text-decoration: underline; font weight: bold;">here</a> to learn more about the architecture of this recommender system. 
    </div>
    """, unsafe_allow_html =True)
    
    st.write("---")
    
    # Centered "About Me" heading.
    st.markdown("<h3 style='text-align: center;'> About Me</h3>", unsafe_allow_html = True)
    st.markdown("""
    <div style = "text-align: justify;">
    <b>Shubham Joshi, Ph.D.</b><br/>
    <i> Applied scientist with expertise in data science and machine learning. In a past life, I was a mathematician who studied beautiful shapes called fractals.</i>
    </div>
    """, unsafe_allow_html = True)
    
    # Links.
    st.markdown("""
    <div style = "display: flex; gap: 15px; justify-content: center; margin-top: 15px;">
        <a href = "https://github.com/shujoshi-git" target="_blank" style="text-decoration: none; color: #66c0f4; font-weight: bold;"> GitHub</a>
        <a href = "https://www.linkedin.com/in/shubham-joshi-ph-d-1625b626b/" target="_blank" style="text-decoration: none; color: #66c0f4; font-weight: bold;"> LinkedIn</a>
        <a href = "mailto:shujoshi.work@gmail.com" style="text-decoration: none; color: #66c0f4; font-weight: bold;"> Email</a>
    </div>
    """, unsafe_allow_html = True)


# UI

st.markdown("<h1 style = 'text-align: center;'>🎮 Steam Games Discovery Engine</h1>", unsafe_allow_html = True)
st.markdown("<h3 style = 'text-align: center; color: #a2b0c4; font-weight: normal; margin-bottom: 25px;'>Tell us what you play. We'll show you what to play next.</h3>", unsafe_allow_html = True)


# Order games to starting with letters first, followed by numbers and then special characters.


def sort_games_key(name):

    name_str = str(name).strip()
    if not name_str:
        return (3, "")  # Empty names at the very bottom.
    
    first_char = name_str[0]
    
    if first_char.isalpha():
        return (0, name_str.lower())
    elif first_char.isdigit():
        return (1, name_str.lower())
    else:
        return (2, name_str.lower())


# Extract unique game names and apply the sorting.
all_available_games = sorted(df_clean['Name'].unique().tolist(), key = sort_games_key)

# Input controls.
with st.container(border = True):
    st.markdown("### 🔍 Select up to **3 games**")
    
    selected_games = st.multiselect("Start typing to search and select games from our database over over 100,000 PC games available on Steam:", options = all_available_games, max_selections = 3, placeholder ="Type to search games...")
    
    # Right-align the generate recs button.
    _, col_btn = st.columns([3, 1])
    with col_btn:
        run_button = st.button("Generate Recommendations", type = "primary", use_container_width= True)

# Execute recommendation algorithm when "Recommend Games" button is clicked.
if run_button:
    if len(selected_games) == 0:
        st.warning("Please select at least one game to get started!")
    else:
        with st.spinner("Baking recommendations..."):
            
            top_paid, top_free, message = recommend_games(selected_games, df_paid_topten, X_paid, df_free_topten, X_free, df_clean, char_vectorizer)
            
            # Display warning message if needed.
            if message:
                if "spelling" in message:
                    st.error(message)
                    st.stop()
                else:
                    st.warning(message)
            else:
                st.success(f"Recommendations successfully generated based on: **{', '.join(selected_games)}**")
            
            # Display 8 paid recommendations.
            st.write("---")
            
            # Center the heading.
            st.markdown("<h2 style = 'text-align: center;'> Top Premium Recommendations</h2>", unsafe_allow_html = True)
            st.write("")  # Adds small visual spacing.
            
            # 2 rows of 4 games each layout.
            cols_paid = st.columns(4)
            for index, (_, row) in enumerate(top_paid.iterrows()):
                with cols_paid[index % 4]:
                    st.markdown(render_game_card(row), unsafe_allow_html = True)
            
            # Display 2 free recommendations.
            st.write("---")
            # Centerthe heading.
            st.markdown("<h2 style = 'text-align: center;'> Top Free-to-Play Recommendations</h2>", unsafe_allow_html = True)
            st.write("")  # Adds small visual spacing
            
            # 2 rows of 2 games each.
            cols_free = st.columns(2)
            for index, (_, row) in enumerate(top_free.iterrows()):
                with cols_free[index % 2]:
                    st.markdown(render_game_card(row), unsafe_allow_html = True)

# Add acknowledgement.

st.write("---")
with st.container():
    st.caption("**Acknowledgements**: This discovery engine is a portfolio project built utilizing publically avaiable video game metadata found on Kaggle and sourced from Steam. It is designed solely for educational, non-commercial purposes. All registered trademarks, logos, and game artwork are the property of their respective owners.")
    