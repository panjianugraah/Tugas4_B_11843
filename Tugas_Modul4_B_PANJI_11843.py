import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import numpy as np
from sklearn.metrics import pairwise_distances
import plotly.graph_objects as go

def scatter(model, model_name, data, new_point, features, color_scale, title):
    clusters = model.fit_predict(data[features])
    data[f"{model_name}_Cluster"] = clusters

    if model_name == "KMeans":
        new_cluster = model.predict(new_point[features].values.reshape(1, -1))
    else:
        distances = pairwise_distances(new_point[features], data[features])  # Updated line
        nearest_index = distances.argmin()
        new_cluster = clusters[nearest_index]

    fig = px.scatter_3d(data, x='Avg_Credit_Limit', y='Total_Credit_Cards', z='Total_visits_online',
                         color=f"{model_name}_Cluster", title=title, color_continuous_scale=color_scale)

    fig.add_trace(
        go.Scatter3d(
            x=[new_point['Avg_Credit_Limit'].values[0]],  # Ensure correct indexing
            y=[new_point['Total_Credit_Cards'].values[0]],  # Ensure correct indexing
            z=[new_point['Total_visits_online'].values[0]],  # Ensure correct indexing
            mode='markers',
            marker=dict(size=10, color='red'),
            name="New Point"
        )
    )

    return fig, new_cluster

st.set_page_config(
    page_title="11843 Unsupervised Learning",
    page_icon="d",
    layout="wide",
    initial_sidebar_state="expanded"
)

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)

    st.markdown("<h1 style='text-align: center;'>Unsupervised Learning 11843</h1>", unsafe_allow_html=True)  # YYYYY diisi dengan nama panggilan
    st.dataframe(input_data)

    model_paths = {
        "AGG_model": r'AGG_model.pkl',
        "KMeans_model": r'KMeans_mode.pkl',
        "DBSCAN_model": 'DBSCAN_model.pkl',
    }

    models = {}
    for model_name, path in model_paths.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[model_name] = pickle.load(f)
        else:
            st.write(f"Model {model_name} tidak ditemukan di path {path}")

    avg_cl = st.sidebar.number_input("Average Credit Limit", min_value=0, max_value=100000)
    tot_cc = st.sidebar.number_input("Total Credit Cards", min_value=0, max_value=18)
    tot_vo = st.sidebar.number_input("Total Visits Online", min_value=0, max_value=150)

    if st.sidebar.button("Predict"):
        features = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_online']

        new_point = pd.DataFrame({
            'Avg_Credit_Limit': [avg_cl],
            'Total_Credit_Cards': [tot_cc],
            'Total_visits_online': [tot_vo]
        })

        cluster_methods = [
            ("KMeans", models["KMeans_model"], "KMeans Clustering", px.colors.sequential.Cividis),
            ("AGG_model", models["AGG_model"], "Agglomerative Clustering", px.colors.sequential.Mint),
            ("DBSCAN_model", models["DBSCAN_model"], "DBSCAN Clustering", px.colors.sequential.Plasma)
        ]

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for i, (model_name, model, title, color_scale) in enumerate(cluster_methods):
            fig, new_cluster = scatter(model, model_name, input_data, new_point, features, color_scale, title)

            with cols[i]:
                st.plotly_chart(fig)
                st.markdown(f"<p style='text-align: center;'>Titik data yang baru masuk ke dalam cluster {new_cluster}</p>", unsafe_allow_html=True)
