
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from textblob import TextBlob
import os

# Configuración
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Parámetros
DELTA_THRESHOLD = 0.5  # Umbral para detectar picos de sentimiento

st.set_page_config(page_title="Sentiment Spike Detector", layout="wide")

st.title("📈 Real-Time Sentiment Spike Detector")
st.markdown("Monitorea sentimiento en tiempo real desde Google News (incluyendo resultados de Reddit) para acciones y criptomonedas.")

# Entrada del usuario
query = st.text_input("🔍 Buscar activo (Ej: AAPL, BTC, TSLA, ETH)", value="Bitcoin")
search_type = st.radio("Tipo de activo", options=["Criptomoneda", "Acción"], horizontal=True)
search_term = f"{query} {'crypto' if search_type == 'Criptomoneda' else 'stock'}"

def fetch_google_news(query):
    url = "https://serpapi.com/search"
    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "q": query,
        "tbm": "nws",
        "num": "20"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error en SerpAPI: {response.status_code}")
        return None

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def process_results(results, source):
    if not results:
        return []

    articles = []
    news_results = results.get("news_results", [])
    for item in news_results:
        title = item.get("title", "")
        link = item.get("link", "")
        date = item.get("date", str(datetime.now()))
        articles.append({
            "title": title,
            "link": link,
            "date": date,
            "source": source
        })
    return articles

if query and SERPAPI_API_KEY:
    with st.spinner("Consultando SerpAPI..."):
        google_main = fetch_google_news(search_term)
        google_reddit = fetch_google_news(f"{search_term} site:reddit.com")

        main_articles = process_results(google_main, "Google News")
        reddit_articles = process_results(google_reddit, "Reddit vía Google News")

        all_articles = main_articles + reddit_articles
        if not all_articles:
            st.warning("No se encontraron artículos.")
        else:
            # Analizar sentimiento
            df = pd.DataFrame(all_articles)
            df["sentiment"] = df["title"].apply(analyze_sentiment)
            df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("datetime")

            # Detección de picos
            df["delta"] = df["sentiment"].diff().fillna(0)
            df["spike"] = df["delta"].abs() > DELTA_THRESHOLD

            # Mostrar gráfico
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df["datetime"],
                y=df["sentiment"],
                mode="lines+markers",
                marker=dict(
                    size=8,
                    color=["red" if s else "blue" for s in df["spike"]],
                    symbol=["star" if s else "circle" for s in df["spike"]]
                ),
                text=df["title"],
                hoverinfo="text+y",
                name="Sentiment"
            ))

            fig.update_layout(
                title=f"Evolución del sentimiento: {query}",
                xaxis_title="Fecha",
                yaxis_title="Sentimiento",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Mostrar titulares que generaron picos
            spike_df = df[df["spike"]]
            if not spike_df.empty:
                st.subheader("⚠️ Titulares que generaron picos de sentimiento")
                for _, row in spike_df.iterrows():
                    st.markdown(f"**[{row['title']}]({row['link']})** ({row['source']}) - Sentimiento: {round(row['sentiment'],2)}")
else:
    st.info("Introduce un término de búsqueda y asegúrate de tener configurada la clave SerpAPI.")
