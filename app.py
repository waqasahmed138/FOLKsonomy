import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ===============================================================
# Load SapBERT model (optimized for biomedical concept matching)
# ===============================================================
@st.cache_resource
def load_model():
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ===============================================================
# Load Diabetes Subset RF2 files from the uploaded location
# ===============================================================
@st.cache_data
def load_rf2_data():
    # Use relative path for the uploaded dataset in Streamlit Cloud
    BASE_DIR = "./diabetes_subset_rf2"  # Folder where the dataset is uploaded
    desc_path = f"{BASE_DIR}/descriptions_diabetes.tsv"
    concept_path = f"{BASE_DIR}/concepts_diabetes.tsv"
    rels_path = f"{BASE_DIR}/relationships_diabetes.tsv"
    base_dir = "/content/drive/MyDrive/diabetes_subset_rf2"  # Update this if needed
    desc_path = f"{base_dir}/descriptions_diabetes.tsv"
    desc = pd.read_csv(desc_path, sep="\t", dtype=str)
    desc = desc[desc["active"] == "1"]
    return desc

descriptions = load_rf2_data()
all_terms = descriptions["term"].dropna().unique().tolist()
concept_ids = descriptions["conceptId"].tolist()

# ===============================================================
# Embed all terms once to calculate embeddings
# ===============================================================
@st.cache_resource
def embed_all_terms(terms):
    vectors = []
    batch_size = 16
    for i in range(0, len(terms), batch_size):
        batch = terms[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:,0,:]
        vectors.append(emb)
    return torch.cat(vectors).cpu().numpy()

term_embeddings = embed_all_terms(all_terms)

# ===============================================================
# Helper: Compute similarity between user input and diabetes terms
# ===============================================================
def get_similarity(input_text):
    inputs = tokenizer([input_text], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    user_vec = outputs.last_hidden_state[:,0,:].cpu().numpy()
    sims = cosine_similarity(user_vec, term_embeddings)[0]
    idx = int(np.argmax(sims))
    return all_terms[idx], concept_ids[idx], float(sims[idx])

# ===============================================================
# Streamlit UI Layout
# ===============================================================
st.set_page_config(page_title="Diabetes Concept Matcher", page_icon="🩸")
st.title("🩸 Diabetes Concept Matcher using SapBERT + SNOMED CT RF2 Subset")
st.markdown("Enter any health-related tag or phrase below to match it to the closest Diabetes concept.")

user_input = st.text_input("💬 Enter a tag (e.g. 'type 2 sugar level high')")

if user_input:
    with st.spinner("Analyzing and matching with SNOMED CT Diabetes subset..."):
        match_term, match_id, score = get_similarity(user_input.lower().strip())

    st.subheader("🔍 Match Result")
    st.write(f"**User Tag:** {user_input}")
    st.write(f"**Closest Match:** {match_term}")
    st.write(f"**Concept ID:** {match_id}")
    st.write(f"**Cosine Similarity:** {score:.3f}")

    # Decision thresholds
    if score >= 0.85:
        st.success("✅ High similarity — existing concept recognized.")
    elif 0.60 <= score < 0.85:
        st.warning("🟡 Medium similarity — potential child concept candidate.")
    else:
        st.error("🔴 Low similarity — likely new or unrelated concept.")


