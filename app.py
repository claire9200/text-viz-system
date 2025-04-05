import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import requests
import fitz  # PyMuPDF
from collections import Counter
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import BytesIO, StringIO
import base64
nltk.download('punkt')

# ------------------------
# Download AIGC Results
# ------------------------
def get_analysis_text_download_link(results_dict, filename="analysis.txt"):
    output = StringIO()
    for key, val in results_dict.items():
        output.write(f"{key}: {val:.4f}\n")
    output.seek(0)
    b64 = base64.b64encode(output.read().encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">📥 Download Analysis Results</a>'
    return href

# ------------------------
# Download Word Cloud
# ------------------------
def get_wordcloud_download_link(wordcloud):
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="wordcloud.png">📥 Download Word Cloud</a>'
    return href

# ------------------------
# Session state setup
# ------------------------
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ------------------------
# Stopwords list
# ------------------------
stopwords = set("""
a about above after again against all am an and any are aren't as at be because been before being below 
between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each 
few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers 
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me 
more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over 
own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs 
them themselves then there there's these they they'd they'll they're they've this those through to too 
under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where 
where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've 
your yours yourself yourselves
""".split())

# ------------------------
# App Title
# ------------------------
st.title("Text Visualizer")

# ------------------------
# Scrape Text from URL
# ------------------------
def scrape_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Error fetching data: {e}"

st.subheader("\U0001F4E5 Optional: Fetch Text from a Webpage")
url_input = st.text_input("Paste a URL (e.g. Wikipedia or news site)")
if st.button("Fetch Text from URL"):
    if url_input:
        with st.spinner("Scraping text..."):
            st.session_state.user_input = scrape_text_from_url(url_input)
            st.success("Text fetched successfully! You can now generate visualizations or run AIGC detection.")
    else:
        st.warning("Please enter a URL!")

# ------------------------
# Upload File
# ------------------------
st.subheader("\U0001F4C2 Optional: Upload a Text or PDF File")
uploaded_file = st.file_uploader("Choose a .txt or .pdf file", type=["txt", "pdf"])
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "txt":
        st.session_state.user_input = uploaded_file.read().decode("utf-8")
        st.success("Text file uploaded and ready!")
    elif file_type == "pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        st.session_state.user_input = text
        st.success("PDF uploaded and extracted!")

# ------------------------
# Text Area
# ------------------------
st.session_state.user_input = st.text_area(
    "Paste your text here manually (or edit text from file/URL)",
    value=st.session_state.user_input,
    height=200
)

# ------------------------
# Word Cloud
# ------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [word for word in text.split() if word not in stopwords]
    return " ".join(tokens)

st.subheader("Word Cloud")
if st.button("Generate Word Cloud"):
    if st.session_state.user_input:
        cleaned = clean_text(st.session_state.user_input)
        wc = WordCloud(width=800, height=400, background_color='white').generate(cleaned)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        st.markdown(get_wordcloud_download_link(wc), unsafe_allow_html=True)
    else:
        st.warning("Please enter some text!")

# ------------------------
# Phrase Network
# ------------------------
st.subheader("Phrase Network Settings")
min_freq = st.slider("Minimum bigram frequency", 1, 20, 5)
k_value = st.slider("Graph spacing (distance between nodes)", 0.1, 2.0, 0.7, 0.1)

def draw_phrase_network(text, min_freq, k_value):
    tokens = nltk.word_tokenize(text.lower())
    bigrams = list(nltk.bigrams(tokens))
    freq_dist = nltk.FreqDist(bigrams)
    filtered_bigrams = [(w1, w2) for (w1, w2), freq in freq_dist.items() if freq >= min_freq]

    if not filtered_bigrams:
        st.warning("Not enough frequent bigrams to display a network. Try adding more text.")
        return

    G = nx.Graph()
    G.add_edges_from(filtered_bigrams)
    pos = nx.spring_layout(G, k=k_value, iterations=100)
    node_sizes = [len(list(G.neighbors(n))) * 150 for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    plt.title("Phrase Network (Filtered & Scaled)")
    st.pyplot(fig)

if st.button("Show Phrase Network"):
    if st.session_state.user_input:
        draw_phrase_network(st.session_state.user_input, min_freq, k_value)
    else:
        st.warning("Please enter some text!")

# ------------------------
# AIGC Detection
# ------------------------
def compute_aigc_features(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text.lower())
    avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    lexical_richness = len(set(words)) / len(words) if words else 0
    phrase_counts = Counter(nltk.ngrams(words, 2))
    repeated = sum(1 for count in phrase_counts.values() if count > 1)
    repetition_score = repeated / len(phrase_counts) if phrase_counts else 0
    sentence_lengths = [len(s.split()) for s in sentences]
    burstiness = np.std(sentence_lengths) if sentence_lengths else 0
    return {
        "Avg Sentence Length": avg_sent_len / 30,
        "Lexical Richness": lexical_richness,
        "Repetition Score": repetition_score,
        "Burstiness": burstiness / 20
    }

def display_radar_chart(features):
    labels = list(features.keys())
    values = list(features.values())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='AIGC Signals',
        line=dict(color='royalblue')
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="AIGC Detection Radar Chart"
    )
    st.plotly_chart(fig)

st.subheader("AIGC Detector")
if st.button("Run AIGC Detection"):
    if st.session_state.user_input:
        with st.spinner("Analyzing..."):
            features = compute_aigc_features(st.session_state.user_input)
            display_radar_chart(features)
            aigc_score = np.mean(list(features.values())) * 100
            st.success(f"Estimated AI-likeness Score: {aigc_score:.1f} / 100")
            st.markdown(get_analysis_text_download_link(features, "aigc_results.txt"), unsafe_allow_html=True)
    else:
        st.warning("Please enter some text!")

# ------------------------
# Topic Modeling (LDA)
# ------------------------
import gensim
from gensim import corpora
from wordcloud import STOPWORDS

def run_topic_modeling(text, num_topics=3):
    tokens = [word for word in nltk.word_tokenize(text.lower()) if word.isalpha() and word not in STOPWORDS]
    if not tokens or len(tokens) < 10:
        st.warning("Not enough content for topic modeling.")
        return
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)

    st.subheader("Topic Modeling (LDA)")
    for idx, topic in lda_model.print_topics():
        st.markdown(f"**Topic {idx+1}:**")

        topic_keywords = topic.split('+')
        words = []
        weights = []

        for kw in topic_keywords[:7]:  # Only show top 7
            weight, word = kw.strip().split('*')
            cleaned_word = word.strip().replace('"', '')
            words.append(cleaned_word)
            weights.append(float(weight))

        # Plot bar chart
        fig, ax = plt.subplots()
        ax.barh(words[::-1], weights[::-1], color='skyblue')
        ax.set_xlabel("Weight")
        ax.set_title(f"Topic {idx+1} Keywords")
        st.pyplot(fig)

st.subheader("Topic Modeling")
if st.button("Run Topic Modeling"):
    if st.session_state.user_input:
        run_topic_modeling(st.session_state.user_input)
    else:
        st.warning("Please enter some text!")

# ------------------------
# Sentiment Analysis (Dual Mode)
# ------------------------

def run_sentiment_vader(text):
    st.subheader("Short Text Mode")
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    st.write(f"**VADER Compound Score:** {compound:.2f} {'😄' if compound > 0.2 else '😐' if -0.2 <= compound <= 0.2 else '😠'}")
    st.write(f"Breakdown: {scores}")
    st.progress((compound + 1) / 2)

def run_sentiment_long(text):
    st.subheader("Long Text Mode")
    analyzer = SentimentIntensityAnalyzer()
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        st.warning("No sentences found for sentiment analysis.")
        return

    scores = [analyzer.polarity_scores(s)['compound'] for s in sentences]
    avg_score = sum(scores) / len(scores)
    
    st.write(f"**Average Sentiment Score:** {avg_score:.2f} {'😄' if avg_score > 0.2 else '😐' if -0.2 <= avg_score <= 0.2 else '😠'}")
    st.write(f"Based on {len(sentences)} sentences")
    st.progress((avg_score + 1) / 2)

st.subheader("Sentiment Analyzer")
st.caption("Use *Short* for reviews or comments. Use *Long* for articles or paragraphs.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Run Sentiment (Short)"):
        if st.session_state.user_input:
            run_sentiment_vader(st.session_state.user_input)
        else:
            st.warning("Please enter some text!")

with col2:
    if st.button("Run Sentiment (Long)"):
        if st.session_state.user_input:
            run_sentiment_long(st.session_state.user_input)
        else:
            st.warning("Please enter some text!")
