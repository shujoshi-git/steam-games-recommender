import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  # Swapped to CountVectorizer!
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Set up a clean, wide page layout
st.set_page_config(
    page_title="Steam Games Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
        /* 1. Completely hide the sidebar collapse/expand trigger button (both open and closed states) */
        button[aria-label="Collapse sidebar"], 
        button[aria-label="Expand sidebar"],
        [data-testid="collapsedControl"],
        .st-emotion-cache-198g9b8, 
        .st-emotion-cache-79elbk {
            display: none !important;
            visibility: hidden !important;
            width: 0px !important;
            height: 0px !important;
        }

        /* 2. Prevent the user from dragging or resizing the sidebar manually */
        [data-testid="stSidebarResizer"] {
            display: none !important;
        }

        /* 3. Custom color override to make the primary button match Steam blue */
        div.stButton > button[kind="primary"] {
            background-color: #1078ff !important;
            color: white !important;
            border: none !important;
        }

        /* 4. Soft blue hover state */
        div.stButton > button[kind="primary"]:hover {
            background-color: #4799ff !important;
        }
        /* 5. Change the background and text color of selected items in multiselect */
        span[data-baseweb="tag"] {
            background-color: #1078ff !important;
            color: white !important;
        }

        /* Ensure the little "X" close icon inside the tags stays readable (white) */
        span[data-baseweb="tag"] svg {
            fill: white !important;
        }
        /* 6. If 3 games are selected, hide the input cursor so the user can't click to open the dropdown */
        div[data-baseweb="select"]:has(span[data-baseweb="tag"]:nth-of-type(3)) input {
            display: none !important;
        }

        /* 7. If 3 games are selected, instantly force-hide any active dropdown/popover menus entirely */
        html:has(span[data-baseweb="tag"]:nth-of-type(3)) div[role="listbox"],
        html:has(span[data-baseweb="tag"]:nth-of-type(3)) div[data-baseweb="popover"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            height: 0px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data():
    """Loads the core game DataFrames and ensures crucial columns exist."""
    df_clean = pd.read_csv('data/games_clean.csv') 
    # Load your top pre-sorted overall popular games for fallback routine
    df_topten = pd.read_csv('data/df_topten.csv')
    df_free_topten = pd.read_csv('data/df_free_topten.csv')
    df_paid_topten = pd.read_csv('data/df_paid_topten.csv')
    
    # Fill empty characteristics strings to prevent vectorization errors
    df_clean['Characteristics'] = df_clean['Characteristics'].fillna('')
    df_topten['Characteristics'] = df_topten['Characteristics'].fillna('')
    df_paid_topten['Characteristics'] = df_paid_topten['Characteristics'].fillna('')
    df_free_topten['Characteristics'] = df_free_topten['Characteristics'].fillna('')
    
    return df_clean, df_paid_topten, df_free_topten, df_topten


@st.cache_resource
def load_matrices(df_clean, df_paid_topten, df_free_topten):
    """
    Fits CountVectorizer and builds the hybrid, feature-stacked candidate matrices 
    matching your notebook's exact count-based logic.
    """
    # 1. Initialize CountVectorizer using a regex pattern that preserves tags 
    # like '2.5d', 'co-op', and 'lgbtq+' as single whole tokens.
    char_vectorizer = CountVectorizer(
        token_pattern=r'[a-zA-Z0-9\.\-\+]+',
        lowercase=True
    )
    char_vectorizer.fit(df_clean['Characteristics'])
    
    # 2. Transform and weight characteristics (50% -> weight factor is sqrt(0.5))
    X_paid_char = char_vectorizer.transform(df_paid_topten['Characteristics']) * np.sqrt(0.5)
    X_free_char = char_vectorizer.transform(df_free_topten['Characteristics']) * np.sqrt(0.5)
    
    # 3. Extract and weight numerical columns (already scaled 0 to 1 in your CSVs)
    votes_paid = (df_paid_topten['Bayesian votes'].values * np.sqrt(0.4)).reshape(-1, 1)
    revenue_paid = (df_paid_topten['Estimated yearly revenue'].values * np.sqrt(0.1)).reshape(-1, 1)
    
    votes_free = (df_free_topten['Bayesian votes'].values * np.sqrt(0.4)).reshape(-1, 1)
    owners_free = (df_free_topten['Estimated owners'].values * np.sqrt(0.1)).reshape(-1, 1)
    
    # 4. Stack features to create the hybrid evaluation space
    X_paid = hstack([X_paid_char, votes_paid, revenue_paid]).tocsr()
    X_free = hstack([X_free_char, votes_free, owners_free]).tocsr()
    
    return char_vectorizer, X_paid, X_free


# Initialize data and models cleanly
df_clean, df_paid_topten, df_free_topten, df_topten = load_data()
char_vectorizer, X_paid, X_free = load_matrices(df_clean, df_paid_topten, df_free_topten)


def recommend_games(input_names, df_paid_topten, X_paid, df_free_topten, X_free, df_master, char_vectorizer, df_topten):
    """
    Recommends 8 paid games and 2 free games using your notebook's hybrid 
    feature-stacked vectorizer architecture.
    """
    # Standardize input titles for robust matching
    input_titles_clean = [title.strip().lower() for title in input_names]
    
    # 1. Locate the input games in the MASTER database
    matched_games = df_master[df_master['Name'].str.strip().str.lower().isin(input_titles_clean)]
    
    # Fallback 1: None of the input games exist on Steam at all
    if matched_games.empty:
        message = "We don't have enough information to recommend games based on the games you input but here are some great games you can check out!"
        fallback_paid = df_paid_topten.sort_values(by='Bayesian votes', ascending=False).head(8)
        fallback_free = df_free_topten.sort_values(by='Bayesian votes', ascending=False).head(2)
        return fallback_paid, fallback_free, message
    
    # Check if matched inputs actually have characteristics
    has_characteristics = matched_games['Characteristics'].fillna('').str.strip() != ""
    
    # Fallback 2: If ALL matched games have missing Characteristics
    if not has_characteristics.any():
        message = "We don't have enough information to recommend games based on the games you input but here are some great games you can check out!"
        fallback_paid = df_paid_topten.sort_values(by='Bayesian votes', ascending=False).head(8)
        fallback_free = df_free_topten.sort_values(by='Bayesian votes', ascending=False).head(2)
        return fallback_paid, fallback_free, message

    # ----------------------------------------------------
    # CORE ROUTINE: BUILD QUERY VECTORS FOR THE INPUT PROFILE
    # ----------------------------------------------------
    profile_games = matched_games[has_characteristics]
    
    # Project characteristics (50%)
    char_sparse_query = char_vectorizer.transform(profile_games['Characteristics'].fillna(''))
    char_weighted_query = char_sparse_query * np.sqrt(0.5)
    
    # Extract numerical matrices (ensuring they are cast to float and any NaN is treated as 0.0)
    votes_query = pd.to_numeric(profile_games['Bayesian votes'], errors='coerce').fillna(0.0).values.reshape(-1, 1)
    revenue_query = pd.to_numeric(profile_games['Estimated yearly revenue'], errors='coerce').fillna(0.0).values.reshape(-1, 1)
    owners_query = pd.to_numeric(profile_games['Estimated owners'], errors='coerce').fillna(0.0).values.reshape(-1, 1)
    
    # A. Build the query vector specifically designed for comparing with PAID games
    votes_weighted_query_paid = (votes_query * np.sqrt(0.4)).reshape(-1, 1)
    revenue_weighted_query_paid = (revenue_query * np.sqrt(0.1)).reshape(-1, 1)
    
    query_matrix_paid = hstack([char_weighted_query, votes_weighted_query_paid, revenue_weighted_query_paid]).tocsr()
    # Safely convert to a dense numpy array before taking the mean
    query_vector_paid = np.asarray(np.mean(query_matrix_paid.toarray(), axis=0)).reshape(1, -1)

    # B. Build the query vector specifically designed for comparing with FREE games
    votes_weighted_query_free = (votes_query * np.sqrt(0.4)).reshape(-1, 1)
    owners_weighted_query_free = (owners_query * np.sqrt(0.1)).reshape(-1, 1)
    
    query_matrix_free = hstack([char_weighted_query, votes_weighted_query_free, owners_weighted_query_free]).tocsr()
    # Safely convert to a dense numpy array before taking the mean
    query_vector_free = np.asarray(np.mean(query_matrix_free.toarray(), axis=0)).reshape(1, -1)

    # ----------------------------------------------------
    # 2. COMPUTE SIMILARITIES AND GET RECOMMENDATIONS
    # ----------------------------------------------------
    
    # Compute similarities against the hybrid PAID target matrix
    paid_similarities = cosine_similarity(query_vector_paid, X_paid).flatten()
    df_paid_results = df_paid_topten.copy()
    df_paid_results['similarity_score'] = paid_similarities
    
    # Filter out inputs to avoid recommending games the user already selected
    df_paid_results = df_paid_results[~df_paid_results['Name'].str.strip().str.lower().isin(input_titles_clean)]
    top_paid = df_paid_results.sort_values(by='similarity_score', ascending=False).head(8)

    # Compute similarities against the hybrid FREE target matrix
    free_similarities = cosine_similarity(query_vector_free, X_free).flatten()
    df_free_results = df_free_topten.copy()
    df_free_results['similarity_score'] = free_similarities
    
    # Filter out inputs
    df_free_results = df_free_results[~df_free_results['Name'].str.strip().str.lower().isin(input_titles_clean)]
    top_free = df_free_results.sort_values(by='similarity_score', ascending=False).head(2)

    return top_paid, top_free, None


def render_game_card(game_row):
    """Generates custom HTML for a clickable grid card featuring the Steam header image and name."""
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


# ------------------------------------------------------------------
# SIDEBAR CONTROLS & INFO
# ------------------------------------------------------------------
with st.sidebar:
    # 1. Centered "About the Engine" Heading
    st.markdown("<h1 style='text-align: center;'>About the Engine</h1>", unsafe_allow_html=True)
    
    # Justified app description
    st.markdown("""
    <div style="text-align: justify;">
    This content-based recommender system maps the underlying <b>thematic feel and mechanics</b> 
    of games using cosine similarity on processed gameplay characteristics and quality metrics. 
    It accepts up to 3 games as <b>input</b> and outputs <b>10</b> high-quality recommendations: 
    8 premium (paid) and 2 free-to-play. Please click <a href="https://github.com/shujoshi-git/steam-games-recommender" target="_blank" style="color: #66c0f4; text-decoration: underline; font-weight: bold;">here</a> to learn more about the architecture of this recommender system. 
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    
    # 2. Centered "About Me" Heading
    st.markdown("<h3 style='text-align: center;'> About Me</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: justify;">
    <b>Shubham Joshi, Ph.D.</b><br/>
    <i>Applied scientist with expertise in data science and machine learning. In a past life, I was a mathematician who studied beautiful shapes called fractals.</i>
    </div>
    """, unsafe_allow_html=True)
    
    # Clean social/portfolio links (centered to align beautifully below the heading)
    st.markdown("""
    <div style="display: flex; gap: 15px; justify-content: center; margin-top: 15px;">
        <a href="https://github.com/shujoshi-git" target="_blank" style="text-decoration: none; color: #66c0f4; font-weight: bold;"> GitHub</a>
        <a href="https://www.linkedin.com/in/shubham-joshi-ph-d-1625b626b/" target="_blank" style="text-decoration: none; color: #66c0f4; font-weight: bold;"> LinkedIn</a>
        <a href="mailto:shujoshi.work@gmail.com" style="text-decoration: none; color: #66c0f4; font-weight: bold;"> Email</a>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# MAIN UI
# ------------------------------------------------------------------

st.markdown("<h1 style='text-align: center;'>🎮 Shubham Joshi's Steam Game Discovery Engine</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #a2b0c4; font-weight: normal; margin-bottom: 25px;'>Tell us what you play. We'll show you what to play next.</h3>", unsafe_allow_html=True)


def sort_games_key(name):
    """
    Groups game titles to start with letters (A-Z), 
    followed by numbers (0-9), and finally special characters.
    """
    name_str = str(name).strip()
    if not name_str:
        return (3, "")  # Empty names at the absolute bottom
    
    first_char = name_str[0]
    
    if first_char.isalpha():
        return (0, name_str.lower())
    elif first_char.isdigit():
        return (1, name_str.lower())
    else:
        return (2, name_str.lower())


# Extract unique game names and apply the custom sorting key
all_available_games = sorted(df_clean['Name'].unique().tolist(), key=sort_games_key)

# Wrap the input controls in a clean, subtle container card
with st.container(border=True):
    st.markdown("### 🔍 Select up to **3 games**")
    
    selected_games = st.multiselect(
        "Start typing to search and select games from our database:",
        options=all_available_games,
        max_selections=3,
        placeholder="Type to search games..."
    )
    
    # Right-align the button by using a wide empty column on the left
    _, col_btn = st.columns([3, 1])
    with col_btn:
        run_button = st.button("Generate Recommendations", type="primary", use_container_width=True)

# Execute core recommendation routine upon click
if run_button:
    if len(selected_games) == 0:
        st.warning("Please select at least one game to get started!")
    else:
        with st.spinner("Analyzing gameplay characteristics and searching the catalogs..."):
            # Execute recommender
            top_paid, top_free, message = recommend_games(
                selected_games, df_paid_topten, X_paid, df_free_topten, X_free, df_clean, char_vectorizer, df_topten
            )
            
            # Display warning/info message if fallback was triggered or spelling check is needed
            if message:
                if "spelling" in message:
                    st.error(message)
                    st.stop()
                else:
                    st.warning(message)
            else:
                st.success(f"Recommendations successfully generated based on: **{', '.join(selected_games)}**")
            
            # --- DISPLAY 8 PAID RECOMMENDATIONS ---
            st.write("---")
            # 3. Centered "Top Premium Recommendations" Heading
            st.markdown("<h2 style='text-align: center;'> Top Premium Recommendations</h2>", unsafe_allow_html=True)
            st.write("")  # Adds small visual spacing
            
            # 4 columns for a clean visual grid layout of 8 games (2 rows of 4)
            cols_paid = st.columns(4)
            for index, (_, row) in enumerate(top_paid.iterrows()):
                with cols_paid[index % 4]:
                    st.markdown(render_game_card(row), unsafe_allow_html=True)
            
            # --- DISPLAY 2 FREE RECOMMENDATIONS ---
            st.write("---")
            # 4. Centered "Top Free-to-Play Recommendations" Heading
            st.markdown("<h2 style='text-align: center;'> Top Free-to-Play Recommendations</h2>", unsafe_allow_html=True)
            st.write("")  # Adds small visual spacing
            
            # 2 columns for a clean visual representation of the 2 free games
            cols_free = st.columns(2)
            for index, (_, row) in enumerate(top_free.iterrows()):
                with cols_free[index % 2]:
                    st.markdown(render_game_card(row), unsafe_allow_html=True)

# ------------------------------------------------------------------
# ACKNOWLEDGEMENT & DISCLAIMER (At the very bottom of the page)
# ------------------------------------------------------------------
st.write("---")
with st.container():
    st.caption(
        "**Acknowledgements:** "
        "This discovery engine is a portfolio project built utilizing publically avaiable video game metadata found on Kaggle and sourced from Steam. "
        "It is designed solely for educational, non-commercial purposes. All registered trademarks, logos, and game artwork are the property "
        "of their respective owners."
    )