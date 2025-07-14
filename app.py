import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import os
import datetime
from textblob import TextBlob

# --- Configuración ---
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
QUERY = st.sidebar.text_input("🔍 Buscar acción o criptomoneda", value="Bitcoin")
NUM_RESULTS = 50
SENTIMENT_SPIKE_THRESHOLD = 0.3  # Umbral para alertas

st.title("📈 Detector de Picos de Sentimiento en Tiempo Real")
st.caption("Datos desde Google News y Reddit usando SerpAPI")

@st.cache_data(show_spinner=False)
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

@st.cache_data(show_spinner=False)
def fetch_news(query):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_news",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": NUM_RESULTS,
        "hl": "en"
    }
    res = requests.get(url, params=params)
    return res.json().get("news_results", [])

@st.cache_data(show_spinner=False)
def fetch_reddit(query):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "reddit",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": NUM_RESULTS
    }
    res = requests.get(url, params=params)
    return res.json().get("organic_results", [])

def build_dataframe():
    news_data = fetch_news(QUERY)
    reddit_data = fetch_reddit(QUERY)

    if not reddit_data:
        st.info("⚠️ No se encontraron resultados recientes en Reddit para esta búsqueda.")

    entries = []
    for item in news_data:
        entries.append({
            "title": item.get("title"),
            "source": "Google News",
            "link": item.get("link"),
            "published": item.get("date") or str(datetime.datetime.now()),
            "sentiment": analyze_sentiment(item.get("title", ""))
        })

    for item in reddit_data:
        entries.append({
            "title": item.get("title"),
            "source": "Reddit",
            "link": item.get("link"),
            "published": item.get("date") or str(datetime.datetime.now()),
            "sentiment": analyze_sentiment(item.get("title", ""))
        })

    df = pd.DataFrame(entries)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published"])
    df = df.sort_values("published")
    return df

df = build_dataframe()

# Mostrar todos los puntos (uno por artículo)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["published"],
    y=df["sentiment"],
    mode="markers",
    marker=dict(color="blue", size=6),
    name="Cada titular"
))

# Calcular promedio de sentimiento por bloques de 5 minutos
df["interval"] = df["published"].dt.floor("5min")
sentiment_over_time = df.groupby("interval")["sentiment"].mean().reset_index()
sentiment_over_time["change"] = sentiment_over_time["sentiment"].diff()

# Detectar picos
spikes = sentiment_over_time[abs(sentiment_over_time["change"]) > SENTIMENT_SPIKE_THRESHOLD]

# --- ALERTAS ---
if not spikes.empty:
    for _, row in spikes.iterrows():
        if row["change"] > 0:
            st.success(f"📈 Sentimiento subió bruscamente a las {row['interval'].strftime('%H:%M')}")
        else:
            st.error(f"📉 Sentimiento bajó bruscamente a las {row['interval'].strftime('%H:%M')}")

# Línea de evolución del sentimiento (promedio)
fig.add_trace(go.Scatter(
    x=sentiment_over_time["interval"],
    y=sentiment_over_time["sentiment"],
    mode="lines+markers",
    name="Promedio 5min",
    line=dict(color="orange")
))

# Anotar titulares en picos
for _, row in spikes.iterrows():
    spike_time = row["interval"]
    related_titles = df[df["interval"] == spike_time].sort_values("sentiment", ascending=False).head(2)
    for _, article in related_titles.iterrows():
        fig.add_trace(go.Scatter(
            x=[spike_time],
            y=[row["sentiment"]],
            mode="markers+text",
            text=[article["title"]],
            textposition="top center",
            marker=dict(size=10, color="red"),
            showlegend=False
        ))

fig.update_layout(title="Evolución del Sentimiento", xaxis_title="Hora", yaxis_title="Sentimiento")
st.plotly_chart(fig, use_container_width=True)

# --- Mostrar tabla de titulares ---
st.subheader("📰 Titulares recientes")
st.dataframe(df[["published", "source", "title", "sentiment"]].sort_values("published", ascending=False), use_container_width=True)
