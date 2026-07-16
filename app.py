import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Set up a clean, wide page layout
st.set_page_config(
    page_title="Steam Games Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)



@st.cache_data
def load_data():
    """Loads the core game DataFrames and ensures crucial columns exist."""
    # Replace with your actual file paths
    df = pd.read_csv('data/games_clean.csv') 
    df_free_topten = pd.read_csv('data/df_free_topten.csv')
    df_paid_topten = pd.read_csv('data/df_paid_topten.csv')
    
    # Fill any empty characteristics or metadata strings to prevent vectorization errors
    df['Characteristics'] = df['Characteristics'].fillna('')
    df_paid_topten['Characteristics'] = df_paid_topten['Characteristics'].fillna('')
    df_free_topten['Characteristics'] = df_free_topten['Characteristics'].fillna('')
    
    return df, df_paid_topten, df_free_topten


@st.cache_resource
def load_matrices():
    """Loads or computes the TF-IDF vectorizer and pre-computed sparse feature matrices."""
    # In production, you can load your pre-saved pickle/joblib files here.
    # For this script, we assume you are initializing them.
    char_vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit on the entire dataset's Characteristics
    char_vectorizer.fit(df['Characteristics'])
    
    # Transform candidate pools
    X_paid = char_vectorizer.transform(df_paid_topten['Characteristics'])
    X_free = char_vectorizer.transform(df_free_topten['Characteristics'])
    
    return char_vectorizer, X_paid, X_free

# Initialize data and models
df, df_paid_topten, df_free_topten = load_data()
char_vectorizer, X_paid, X_free = load_matrices()




def recommend_games(input_names, df_paid_topten, X_paid, df_free_topten, X_free, df, char_vectorizer):
    """
    Computes cosine similarity between user input games and candidate pools.
    Returns top_paid (DataFrame), top_free (DataFrame), and a warning message (if any).
    """
    # 1. Clean input strings and filter out names not present in our vocabulary
    valid_inputs = [name.strip() for name in input_names if name.strip() in df['Name'].values]
    
    if not valid_inputs:
        return None, None, "None of the inputted games were found in our database. Please check your spelling!"
        
    # 2. Extract characteristics of input games and combine them into a single query vector
    input_rows = df[df['Name'].isin(valid_inputs)]
    combined_characteristics = " ".join(input_rows['Characteristics'].tolist())
    
    # Transform query to TF-IDF space
    query_vector = char_vectorizer.transform([combined_characteristics])
    
    # 3. Calculate similarities against Paid and Free candidate pools
    similarity_paid = cosine_similarity(query_vector, X_paid).flatten()
    similarity_free = cosine_similarity(query_vector, X_free).flatten()
    
    # Add similarity score as a temporary column for sorting
    df_paid_copy = df_paid_topten.copy()
    df_free_copy = df_free_topten.copy()
    
    df_paid_copy['similarity'] = similarity_paid
    df_free_copy['similarity'] = similarity_free
    
    # Filter out the input games themselves from the recommendations to avoid regurgitation
    df_paid_copy = df_paid_copy[~df_paid_copy['Name'].isin(valid_inputs)]
    df_free_copy = df_free_copy[~df_free_copy['Name'].isin(valid_inputs)]
    
    # 4. Extract top 5 paid and top 5 free recommendations
    top_paid = df_paid_copy.sort_values(by='similarity', ascending=False).head(5)
    top_free = df_free_copy.sort_values(by='similarity', ascending=False).head(5)
    
    return top_paid, top_free, None



def render_game_card(game_row):
    """Generates custom HTML for a clickable grid card featuring the Steam header image and name."""
    name = game_row['Name']
    website = game_row['Website']
    header_image = game_row['Header image']
    
    # Fallback to a placeholder if the header image is missing
    if pd.isna(header_image) or str(header_image).strip() == "":
        header_image = "https://via.placeholder.com/460x215.png?text=No+Image+Available"
        
    # Fallback if website is missing
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
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/83/Steam_icon_logo.svg", width=80)
    st.title("About the Engine")
    st.markdown("""
    This hybrid recommender maps the underlying **thematic feel and mechanics** of games using a TF-IDF vectorizer trained on processed gameplay characteristics. 
    
    To encourage **serendipitous discovery** and bypass obvious filter bubbles, corporate metadata (like Publishers and Developers) has been stripped out.
    """)
    st.write("---")
    st.info("💡 **Tip:** Try mixing different genres (e.g., a puzzle game and a racing game) to see how the engine synthesizes their characteristics!")

# ------------------------------------------------------------------
# MAIN UI
# ------------------------------------------------------------------
st.title("🎮 Steam Game Discovery Engine")
st.subheader("Map the 'feel' of your favorite games to your next obsession")

# Input field: Multi-select dropdown paired with custom manual text input
all_available_games = sorted(df['Name'].unique().tolist())

st.write("### Step 1: Input Your Favorite Games")
selected_games = st.multiselect(
    "Search and select up to 3 games from our database:",
    options=all_available_games,
    max_selections=3,
    placeholder="Type to search games..."
)

# Run Button
if st.button("Generate Recommendations 🚀", type="primary"):
    if len(selected_games) == 0:
        st.warning("Please select at least one game to get started!")
    else:
        with st.spinner("Analyzing gameplay characteristics and searching the catalogs..."):
            # Execute recommender
            top_paid, top_free, error_msg = recommend_games(
                selected_games, df_paid_topten, X_paid, df_free_topten, X_free, df, char_vectorizer
            )
            
            if error_msg:
                st.error(error_msg)
            else:
                st.success(f"Recommendations successfully generated based on: **{', '.join(selected_games)}**")
                
                # --- DISPLAY PAID RECOMMENDATIONS ---
                st.write("---")
                st.markdown("## 💎 Top Premium Recommendations")
                
                # Create a 5-column layout for the 5 paid games
                cols_paid = st.columns(5)
                for index, (_, row) in enumerate(top_paid.iterrows()):
                    with cols_paid[index % 5]:
                        st.markdown(render_game_card(row), unsafe_allow_html=True)
                
                # --- DISPLAY FREE RECOMMENDATIONS ---
                st.write("---")
                st.markdown("## 🎁 Top Free-to-Play Recommendations")
                
                # Create a 5-column layout for the 5 free games
                cols_free = st.columns(5)
                for index, (_, row) in enumerate(top_free.iterrows()):
                    with cols_free[index % 5]:
                        st.markdown(render_game_card(row), unsafe_allow_html=True)