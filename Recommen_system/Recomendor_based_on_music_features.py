from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import streamlit.components.v1 as components
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import streamlit as st
st.set_page_config(page_title="Song Recommendation", layout="wide")


@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("../track_with_year_and_genre.csv")
    df['genres'] = df.genres.apply(
        lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df


genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
               'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()


def n_neighbors_uri_audio(start_year, end_year, test_feat):
    genre_data = exploded_track_df[(exploded_track_df["release_year"] >= start_year) & (
        exploded_track_df["release_year"] <= end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:100]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(
        genre_data), return_distance=False)[0]

    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios


title = "Song Recommendation Engine"
st.title(title)

st.write("First of all, welcome! This is the place where you can customize what you want to listen to based on genre and several key audio features. Try playing around with different settings and listen to the songs recommended by our system!")
st.markdown("##")

with st.container():
    col1, col2, col3, col4 = st.columns((2, 0.5, 0.5, 0.5))
    with col1:
        st.markdown("***Choose features to customize:***")
        start_year, end_year = st.slider(
            'Select the year range',
            1990, 2019, (2015, 2019)
        )
        acousticness = st.slider(
            'Acousticness',
            0.0, 1.0, 0.5)
        danceability = st.slider(
            'Danceability',
            0.0, 1.0, 0.5)
        energy = st.slider(
            'Energy',
            0.0, 1.0, 0.5)
        instrumentalness = st.slider(
            'Instrumentalness',
            0.0, 1.0, 0.0)
        valence = st.slider(
            'Valence',
            0.0, 1.0, 0.45)
        tempo = st.slider(
            'Tempo',
            0.0, 244.0, 118.0)

test_feat = [acousticness, danceability,
             energy, instrumentalness, valence, tempo]
tracks_per_page = 6

if st.button('Add data'):
    uris, audios = n_neighbors_uri_audio(
        start_year, end_year, test_feat)

    # Create an empty dataframe with some columns
    df = pd.DataFrame(columns=['Track name', 'Artist name', 'release_date'])
    for uri in uris:
        # Replace these values with your own Spotify client ID and secret
        client_id = '085bbceb8373490490fd9e97ec126384'
        client_secret = 'b30faec8d58c409ab2984eacb18834b4'

        # Initialize the Spotify client credentials
        client_credentials_manager = SpotifyClientCredentials(
            client_id, client_secret)
        spotify = spotipy.Spotify(
            client_credentials_manager=client_credentials_manager)

        # Replace this with the track URI you want to search for
        track_uri = uri

        # Search for the track using its URI
        results = spotify.track(track_uri)

        new_data = {'Track name': results["name"],
                    'Artist name': results["artists"][0]["name"],
                    'release_date': results['album']['release_date']}

        df = df.append(new_data, ignore_index=True)

    # Print the table using streamlit
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    st.markdown(
        f"""
    <div style='width: 50%'>
        {df.to_html()}
    </div>
    """,
        unsafe_allow_html=True
    )

    st.write("No songs left to recommend")
