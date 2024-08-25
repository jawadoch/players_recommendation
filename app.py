##### Streamlit Python Application #####
# used libraries 
import os
import json
import pandas as pd
import plotly.graph_objects as go
import re
from typing import List
# Solr library
from SolrClient import SolrClient
# Bing library for image automation
from bing_image_urls import bing_image_urls
# Streamlit libraries
import streamlit as st 
from streamlit_searchbox import st_searchbox
# Cosine similarity libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
# Set page configuration
st.set_page_config(page_title="Player Scouting Recommendation System", page_icon="⚽", layout="wide")
st.markdown("<h1 style='text-align: center;'>⚽🔍 Player Scouting Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Scout, Recommend, Elevate Your Team's Game - 👀 </p>", unsafe_allow_html=True)
# Initialize Solr client
solr = SolrClient('http://localhost:8983/solr')
# Initialize session state for maintaining state across page reloads
if 'expanded' not in st.session_state:
    st.session_state.expanded = True

if 'choice' not in st.session_state:
    st.session_state.choice = None

# Function to search Solr and return player suggestions
def search_solr(searchterm: str) -> List[str]:
    if searchterm:
        # Query Solr for player names containing the search term
        res = solr.query('FootballStatsCore', {
            'q': f'Player:*{searchterm}*',
            'fl': 'Player',
            'rows': 10,
        })
        result = res.docs

        # If results are found, process them
        if result:
            df_p = pd.DataFrame(result)
            df_p['Player'] = df_p['Player'].apply(lambda x: x[0])
            return df_p['Player'].tolist()
        else:
            return []

# Streamlit search box for player search
selected_value = st_searchbox(
    search_solr,
    key="solr_searchbox",
    placeholder="🔍 Search a Football Player"
)

# Save the selected value in session state
st.session_state.choice = selected_value
choice = st.session_state.choice

################### Organic result ###########################
if choice:
    # Query Solr to retrieve all players data
    res = solr.query('FootballStatsCore', {
        'q': '*:*',
        'fl': '*',
        'rows': 100000,  # Set this value to any large number that exceeds the total number of documents
    })

    result = res.docs

    # Create a DataFrame from the Solr query results
    df_player = pd.DataFrame(result)
    
    # Extract column names from the JSON result
    columns_to_process = list(df_player.columns)

    # Process columns in the DataFrame
    for column in columns_to_process:
        if isinstance(df_player[column][0], (list, dict)):
            df_player[column] = df_player[column].apply(lambda x: x[0] if isinstance(x, list) else (x if isinstance(x, dict) else x))

    # Define columns to drop from the DataFrame
    columns_to_drop = ['id', '_version_']
    df_player = df_player.drop(columns=columns_to_drop)

    # Create a normalized copy of the player DataFrame
    df_player_norm = df_player.copy()

    # Define a custom mapping for the 'Pos' column
    custom_mapping = {
        'GK': 1,
        'DF,FW': 4,
        'MF,FW': 8,
        'DF': 2,
        'DF,MF': 3,
        'MF,DF': 5,
        'MF': 6,
        'FW,DF': 7,
        'FW,MF': 9,
        'FW': 10
    }
    # Apply the custom mapping to the 'Pos' column
    df_player_norm['Pos'] = df_player_norm['Pos'].map(custom_mapping)    
    # Select a subset of features for analysis
    selected_features = ['Pos', 'Age', 'Int',
       'Clr', 'KP', 'PPA', 'CrsPA', 'PrgP', 'Playing_Time_MP',
       'Performance_Gls', 'Performance_Ast', 'Performance_G_A',
       'Performance_G-PK', 'Performance_Fls', 'Performance_Fld',
       'Performance_Crs', 'Performance_Recov', 'Expected_xG', 'Expected_npxG', 'Expected_xAG',
       'Expected_xA', 'Expected_A-xAG', 'Expected_G-xG', 'Expected_np_G-xG',
       'Progression_PrgC', 'Progression_PrgP', 'Progression_PrgR',
       'Tackles_Tkl', 'Tackles_TklW', 'Tackles_Def_3rd', 'Tackles_Mid_3rd',
       'Tackles_Att_3rd', 'Challenges_Att', 'Challenges_Tkl_',
       'Challenges_Lost', 'Blocks_Blocks', 'Blocks_Sh', 'Blocks_Pass',
       'Standard_Sh', 'Standard_SoT', 'Standard_SoT_', 'Standard_Sh_90', 'Standard_Dist', 'Standard_FK',
       'Performance_GA', 'Performance_SoTA', 'Performance_Saves',
       'Performance_Save_', 'Performance_CS', 'Performance_CS_',
       'Penalty_Kicks_PKatt', 'Penalty_Kicks_Save_', 'SCA_SCA',
       'GCA_GCA', 
       'Aerial_Duels_Won', 'Aerial_Duels_Lost', 'Aerial_Duels_Won_',
       'Total_Cmp', 'Total_Att', 'Total_Cmp_', 'Total_TotDist',
       'Total_PrgDist', '1_3'
    ]

    ####################### Cosine Similarity #######################################
    # Normalization using Min-Max scaling
    scaler = MinMaxScaler()
    df_player_norm[selected_features] = scaler.fit_transform(df_player_norm[selected_features])

    # Calculate cosine similarity between players based on selected features
    similarity = cosine_similarity(df_player_norm[selected_features])

    # Find the Rk associated with the selected player's name
    index_player = df_player.loc[df_player['Player'] == choice, 'Rk'].values[0]

    # Calculate similarity scores and sort them in descending order
    similarity_score = list(enumerate(similarity[index_player]))
    similar_players = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Create a list to store data of similar players
    similar_players_data = []

    # Loop to extract information from similar players
    for player in similar_players[0:11]:  # Exclude the first player (self)
        index = player[0]
        similarity_percentage = f"{player[1] * 100:.2f}%"  # Convert to percentage
        player_records = df_player[df_player['Rk'] == index]
        if not player_records.empty:
            player_data = player_records.iloc[0]  # Get the first row (there should be only one)
            player_data = player_data.to_dict()  # Convert Series to dict for easier handling
            player_data['Similarity'] = similarity_percentage 
            similar_players_data.append(player_data)

    # Create a DataFrame from the data of similar players
    similar_players_df = pd.DataFrame(similar_players_data)

    ########################## Analytics of the player chosen ##########################
    url_player = bing_image_urls(choice+ " "+df_player.loc[df_player['Player'] == choice, 'Squad'].iloc[0]+" 2023", limit=1, )[0]

    with st.expander("Features of The Player selected", expanded=True):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(choice)
            st.image(url_player, width=356)

        with col2:
            st.caption("📄 Information of Player")
            col_1, col_2, col_3 = st.columns(3)

            with col_1:
                st.metric("Nation", df_player.loc[df_player['Player'] == choice, 'Nation'].iloc[0], None)
                st.metric("Position", df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0], None)

            with col_2:
                st.metric("Born",df_player.loc[df_player['Player'] == choice, 'Born'].iloc[0],None)
                st.metric("Match Played",df_player.loc[df_player['Player'] == choice, 'Playing_Time_MP'].iloc[0],None, help="In 2022/2023")

            with col_3:
                st.metric("Age",df_player.loc[df_player['Player'] == choice, 'Age'].iloc[0],None)

            st.metric(f"🏆 League: {df_player.loc[df_player['Player'] == choice, 'Comp'].iloc[0]}",df_player.loc[df_player['Player'] == choice, 'Squad'].iloc[0],None, help="In 2022/2023")

        with col3:
            st.caption("⚽ Information target of Player")
            #GK
            if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "GK":
                col_1, col_2 = st.columns(2)

                with col_1:
                    st.metric("Saves", df_player.loc[df_player['Player'] == choice, 'Performance_Saves'].iloc[0], None)
                    st.metric("Clean Sheet", df_player.loc[df_player['Player'] == choice, 'Performance_CS'].iloc[0], None)

                with col_2:
                    st.metric("Goals Against",df_player.loc[df_player['Player'] == choice, 'Performance_GA'].iloc[0],None)
                    st.metric("ShoTA",df_player.loc[df_player['Player'] == choice, 'Performance_SoTA'].iloc[0],None)
            
            # DF
            if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF,MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF,FW":
                col_1, col_2, col_3 = st.columns(3)

                with col_1:
                    st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance_Ast'].iloc[0], None)
                    st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance_Gls'].iloc[0], None)

                with col_2:
                    st.metric("Aerial Duel",df_player.loc[df_player['Player'] == choice, 'Aerial_Duels_Won'].iloc[0],None)
                    st.metric("Tackle",df_player.loc[df_player['Player'] == choice, 'Tackles_TklW'].iloc[0],None, help="In 2022/2023")

                with col_3:
                    st.metric("Interception",df_player.loc[df_player['Player'] == choice, 'Int'].iloc[0],None)
                    st.metric("Key Passage",df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0],None)

            # MF
            if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF,DF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF,FW":
                col_1, col_2, col_3 = st.columns(3)

                with col_1:
                    st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance_Ast'].iloc[0], None)
                    st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance_Gls'].iloc[0], None)
                    st.metric("Aerial Duel",df_player.loc[df_player['Player'] == choice, 'Aerial_Duels_Won'].iloc[0],None)

                with col_2:
                    st.metric("GCA",df_player.loc[df_player['Player'] == choice, 'GCA_GCA'].iloc[0],None)
                    st.metric("Progressive PrgP",df_player.loc[df_player['Player'] == choice, 'Progression_PrgP'].iloc[0],None, help="In 2022/2023")

                with col_3:
                    st.metric("SCA",df_player.loc[df_player['Player'] == choice, 'SCA_SCA'].iloc[0],None)
                    st.metric("Key Passage",df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0],None)
            
            # FW
            if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW,MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW,DF":
                col_1, col_2, col_3 = st.columns(3) 

                with col_1:
                    st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance_Ast'].iloc[0], None)
                    st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance_Gls'].iloc[0], None)
                    st.metric("Aerial Duel",df_player.loc[df_player['Player'] == choice, 'Aerial_Duels_Won'].iloc[0],None)

                with col_2:
                    st.metric("SCA",df_player.loc[df_player['Player'] == choice, 'SCA_SCA'].iloc[0],None)
                    st.metric("xG",df_player.loc[df_player['Player'] == choice, 'Expected_xG'].iloc[0],None)
                    st.metric("xAG",df_player.loc[df_player['Player'] == choice, 'Expected_xAG'].iloc[0],None, help="In 2022/2023")
                    

                with col_3:
                    st.metric("GCA",df_player.loc[df_player['Player'] == choice, 'GCA_GCA'].iloc[0],None)
                    st.metric("Key Passage",df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0],None)
                                
    ################# Radar and Rank ######################### 
    col1, col2 = st.columns([1.2, 2])

    with col1:
        ###### Similar Players Component ###############
        st.subheader(f'Top 10 Similar Players to {choice}')
        selected_columns = ["Player", "Nation", "Squad", "Pos", "Age", "Similarity"]
        st.dataframe(similar_players_df.iloc[1:][selected_columns], hide_index=True, use_container_width=True)

    with col2:
        ###### Radar Analytics #########################
        categories = ['Performance_Gls', 'Performance_Ast', 'KP', 'GCA_GCA','Aerial_Duels_Won', 'Int', 'Tackles_TklW', 'Performance_Saves', 'Performance_CS', 'Performance_GA','Performance_SoTA']
        selected_players = similar_players_df.head(2)
        most_similar_player=selected_players.iloc[1]['Player']
        st.subheader(f'{most_similar_player} is the most similar player to {choice}.')
        fig = go.Figure()

        for index, player_row in selected_players.iterrows():
            player_name = player_row['Player']
            values = [player_row[col] for col in categories]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=player_name
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )
            ),
            showlegend=True,  
            legend=dict(
                orientation="v", 
                yanchor="top",  
                y=1,  
                xanchor="left",  
                x=1.02,  
            ),
            width=750,  
            height=450  
        )

        st.plotly_chart(fig, use_container_width=True)
