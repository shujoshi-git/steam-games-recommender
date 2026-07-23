**Live App Demo**: [Steam Games Recommender](https://shujoshi-steam-recommender.streamlit.app/).

# **Video Game Recommender, Business Analytics & Web App**

**Goal**: The goal of this project is to build an end-to-end content-based Recommender System (RS) for video games using a dataset of over 120,000 PC video games available on Steam. Along the way, we answer several insightful business questions. Finally, we deploy the recommender system as an app using Streamlit. 

**Logistics**:

* [notebooks/data_business.ipynb](./notebooks/data_business.ipynb): Walks through the entire data science pipeline: from raw metadata cleaning and exploratory data analysis to answering key business insights.

* [notebooks/recommender.ipynb](./notebooks/recommender.ipynb): Covers the design and execution of the RS architecture, followed by both qualitative evaluation (relevance and serendipity) and quantitative evaluation (using Recall, ILD and novelty.)

* [app.py](/app.py): Deploys the production ready recommendation algorithm as a web application to Streamlit. 

* Interactive Web App: You can test the live recommender here: [Steam Games Recommender](https://shujoshi-steam-recommender.streamlit.app/).

**High-level overview of the Recommender System**:

1. The user will input three games and the RS will output 10 similar games, 2 of which are free-to-play.
2. The features of the input games will be converted to a similarity matrix and its distance to all games in a subset of df_topten will be calculated using cosine similarity.
3. Games to be recommended will be ranked by cosine similarity (top games have the largest similarity score).
4. The top 8 paid games and the top 2 free-to-play games will be output as the recommendations.

Some salient features of the Recommender System:

* **Under the hood**: We construct two recommender systems: one will recommend 8 paid games and the other, 2 free-to-play games. Constructing two recommender systems is necessary since the features used to recommend paid games are not the same as the ones used to recommend free games. 

* **Cold Start Mitigation**: If all three input games exist in our dataset but are missing Characteristics data, the recommender will print out a message along with a curated list of top games to recommend. 

* **Serendipity vs Redundancy**: The RS is constructed so that recommendations are not necessarily other titles in the same franchise as the input. This ensures that titles from other franchises surface up and could act as serendipitous discoveries for the user. 
