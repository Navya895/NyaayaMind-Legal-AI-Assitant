import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import pdfplumber
import textwrap
import requests
from bs4 import BeautifulSoup
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline
from transformers import pipeline
import math
import re
from collections import Counter


# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="⚖️ NyaayaMind", layout="wide", page_icon="⚖️")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.image(r"C:\Users\Navya\Desktop\navya lap\College\SEM5\NLP\nyaayamind.png", width=120)
st.sidebar.markdown("### ⚖️ **NyaayaMind**")
st.sidebar.write("Empowering legal research with AI.")
st.sidebar.markdown("[📜 Open Indian Constitution](https://legislative.gov.in/constitution-of-india)")
st.sidebar.write("---")
st.sidebar.markdown("👩‍💻 *Developed by Navya Singh and Reshma Patil*")

st.title("⚖️ NyaayaMind: Legal Case Assistant")
st.caption("Analyze, Summarize & Classify Legal Judgments using AI")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_summarizer():
    # Use Pegasus model fine-tuned for legal summarization
    try:
        return pipeline("summarization", model="nsi319/legal-pegasus", tokenizer="nsi319/legal-pegasus")
    except:
        # fallback
        return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_classifier():
    model_path = r"C:\Users\Navya\Desktop\navya lap\College\SEM5\NLP\Project\inlegal_bert"
    return pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=-1
    )
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

summarizer = load_summarizer()
classifier = load_classifier()
embedder = load_embedder()

# -----------------------------
# Corpus and Labels
# -----------------------------
label_names = [
    "Civil", "Corporate", "Family", "Taxation",
    "Criminal", "Labour", "Property", "Other"
]

corpus = [
    "Case about corporate fraud and taxation issues.",
    "Case related to family dispute and divorce settlement.",
    "Case involving labour rights and employment law violations.",
    "Case concerning ownership dispute over land and property.",
    "Case involving constitutional validity of taxation reforms."
]
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

summarizer = load_summarizer()
classifier = load_classifier()
embedder = load_embedder()
# -----------------------------
# Web Scraper (Indian Kanoon)
# -----------------------------
import re
from collections import Counter


def fetch_similar_cases_from_web(text):
    """
    Fetch top 5 similar cases from Indian Kanoon by extracting keywords from summary text.
    """

    # Clean text and extract keywords
    words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())  # take words with >=4 letters
    stop_words = {"case", "court", "judge", "filed", "appeal", "petitioner", "respondent",
                  "section", "act", "india", "indian", "order", "proceeding", "judgment"}
    filtered_words = [w for w in words if w not in stop_words]

    # Pick top frequent keywords
    top_keywords = [word for word, _ in Counter(filtered_words).most_common(8)]
    if not top_keywords:
        top_keywords = words[:8]
    query = "+".join(top_keywords)

    url = f"https://indiankanoon.org/search/?formInput={query}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all("a", class_="result_title")
        case_titles = [r.get_text(strip=True) for r in results[:5]]

        return case_titles if case_titles else [f"No similar cases found for query: {', '.join(top_keywords)}"]

    except Exception as e:
        return [f"Error fetching cases: {e}"]



# -----------------------------
# File Upload
# -----------------------------
st.markdown("### 📁 Upload your Legal Document")
uploaded_file = st.file_uploader("Upload a PDF or TXT case file", type=["txt", "pdf"])

case_text = ""
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    case_text += text + "\n"
    else:
        case_text = uploaded_file.read().decode("utf-8")

    st.success("✅ Case file uploaded successfully!")

# -----------------------------
# Helper Functions
# -----------------------------
def generate_legal_summary(text):
    chunk_size = 1500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks[:5]:
        partial_summary = summarizer(chunk, max_length=256, min_length=80, do_sample=False)[0]['summary_text']
        summaries.append(partial_summary)

    structured_summary = f"""
    ### 🧾 **Facts of the Case**
    {textwrap.fill(summaries[0] if summaries else "Details unavailable.", 120)}

    ### ⚖️ **Issues Raised**
    {textwrap.fill(summaries[1] if len(summaries) > 1 else summaries[0], 120)}

    ### 🧩 **Court’s Observations**
    {textwrap.fill(summaries[2] if len(summaries) > 2 else summaries[-1], 120)}

    ### 🏛️ **Final Judgment**
    {textwrap.fill(summaries[-1] if len(summaries) > 3 else "The court rendered its final decision based on evidence and submissions of both parties.", 120)}
    """
    return structured_summary

def predict_domain(text):
    chunk_size = 512
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    scores = torch.zeros(len(label_names))

    for chunk in chunks[:10]:
        preds = classifier(chunk)

        # ✅ Handle both possible output types
        if isinstance(preds, list):
            preds = preds[0]  # get the first prediction dict

        if not isinstance(preds, dict):
            continue  # skip invalid predictions

        label = preds.get('label', None)
        score = preds.get('score', 0.0)
        if not label:
            continue

        try:
            idx = int(label.split("_")[-1])
            scores[idx] += score
        except Exception:
            pass

    # ✅ Fallback if no scores found
    if scores.sum() == 0:
        return "Other", 0.0

    domain_idx = torch.argmax(scores).item()
    confidence = float(scores[domain_idx] / scores.sum())
    return label_names[domain_idx], confidence


# -----------------------------
# Tabs for Functionalities
# -----------------------------
if case_text:
    tab1, tab2, tab3 = st.tabs(["📝 Detailed Summarization", "🏷️ Domain Classification", "🔍 Similar Case Finder"])

    # --- 📝 Summarization Tab ---
    # --- 📝 Summarization Tab ---
    with tab1:
        st.header("📖 Case Summary Generator")
        st.info("Generates a structured summary covering **facts, issues, observations, and judgment.**")

        if st.button("Generate Detailed Summary", key="summarize"):
            with st.spinner("🔍 Summarizing your case file..."):
                summary = generate_legal_summary(case_text)
                st.success("✅ Summary Generated!")

                # Custom HTML styling for a professional look
                st.markdown("""
                <style>
                .section-box {
                    background-color: #111827;
                    padding: 15px 20px;
                    border-radius: 12px;
                    margin-bottom: 15px;
                    border: 1px solid #2d3748;
                }
                .section-title {
                    font-size: 18px;
                    font-weight: 600;
                    color: #93c5fd;
                    margin-bottom: 10px;
                }
                .section-content {
                    color: #e5e7eb;
                    line-height: 1.6;
                    font-size: 15px;
                    text-align: justify;
                    max-height: 200px;
                    overflow-y: auto;
                }
                </style>
                """, unsafe_allow_html=True)

                # Split structured summary text
                sections = summary.split("### ")
                for sec in sections:
                    if not sec.strip():
                        continue
                    if "Facts of the Case" in sec:
                        st.markdown(f"""
                        <div class='section-box'>
                            <div class='section-title'>📘 Facts of the Case</div>
                            <div class='section-content'>{sec.split('Facts of the Case')[-1].strip()}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    elif "Issues Raised" in sec:
                        st.markdown(f"""
                        <div class='section-box'>
                            <div class='section-title'>⚖️ Issues Raised</div>
                            <div class='section-content'>{sec.split('Issues Raised')[-1].strip()}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    elif "Court’s Observations" in sec:
                        st.markdown(f"""
                        <div class='section-box'>
                            <div class='section-title'>👁️ Court’s Observations</div>
                            <div class='section-content'>{sec.split('Court’s Observations')[-1].strip()}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    elif "Final Judgment" in sec:
                        st.markdown(f"""
                        <div class='section-box'>
                            <div class='section-title'>🏛️ Final Judgment</div>
                            <div class='section-content'>{sec.split('Final Judgment')[-1].strip()}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # --- 🏷️ Classification Tab ---
    with tab2:
        st.header("🏷️ Case Domain Classification")
        st.info("Predicts whether the uploaded case belongs to Civil, Criminal, Corporate, etc.")
        if st.button("Predict Case Domain", key="classify"):
            with st.spinner("🔎 Classifying domain..."):
                result = classifier(case_text[:1024])[0]

                try:
                    label_index = int(result['label'].split("_")[-1])
                    domain = label_names[label_index]
                except:
                    domain = result['label']

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Domain", domain)
                with col2:
                    st.metric("Confidence Score", f"{result['score']:.4f}")


    # --- 🔍 Similar Case Finder Tab ---
    with tab3:
        st.header("🔍 Find Similar Cases")
        st.info("Finds cases with similar facts or legal themes both locally and online.")

        if st.button("Find Similar Cases", key="similar"):
                with st.spinner("🔎 Finding similar cases..."):

                    # 1️⃣ Predict domain first
                    domain_result = classifier(case_text[:1024])[0]
                    try:
                        label_index = int(domain_result['label'].split("_")[-1])
                        predicted_domain = label_names[label_index]
                    except:
                        predicted_domain = domain_result['label']

                    st.info(f"Predicted Domain for Similarity Search: **{predicted_domain}**")

                    # 2️⃣ Simple domain corpora (expandable)
                    domain_corpora = {
                        "Civil": [
                            "Dispute related to property and land ownership.",
                            "Civil appeal concerning breach of contract.",
                            "Damages and tort liability case."
                        ],
                        "Corporate": [
                            "Corporate fraud involving company directors.",
                            "Taxation issues for private limited company.",
                            "Mergers and acquisitions dispute."
                        ],
                        "Criminal": [
                            "Murder and theft under IPC sections.",
                            "Criminal conspiracy and assault case.",
                            "Narcotics Act and criminal misconduct."
                        ]
                    }
                    domain_corpus = domain_corpora.get(predicted_domain, corpus)

                    # 3️⃣ Compute similarity within same-domain corpus
                    query_embedding = embedder.encode([case_text], convert_to_tensor=True)
                    corpus_embeddings = embedder.encode(domain_corpus, convert_to_tensor=True)
                    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)
                    top_results = torch.topk(cos_scores, k=min(3, len(domain_corpus)))

                    st.subheader("📚 Top 3 Similar Cases (Local Corpus)")
                    for i, idx in enumerate(top_results.indices[0]):
                        st.markdown(f"**{i+1}.** {domain_corpus[idx]}  \n🔗 *Cosine Similarity:* {float(top_results.values[0][i]):.4f}")

                    # 4️⃣ Fetch online similar cases
                    st.subheader("🌐 Top Similar Cases from Indian Kanoon")
                    summary = summarizer(case_text[:1500], max_length=200, min_length=60, do_sample=False)[0]['summary_text']
                    web_results = fetch_similar_cases_from_web(f"{predicted_domain} {summary}")
                    for i, title in enumerate(web_results):
                        st.markdown(f"**{i+1}.** [{title}](https://indiankanoon.org)")
                        