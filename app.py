import streamlit as st
import pandas as pd
import zipfile
import os
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

# ---------------------------------------------------------------
# Load SapBERT model (optimized for biomedical concept matching)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------------------------------------------
# Function to extract ZIP file if not already extracted
# ---------------------------------------------------------------
def extract_zip(zip_file, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        st.write("✅ ZIP file extracted.")
    else:
        st.write("📂 ZIP already extracted.")

# ---------------------------------------------------------------
# Download and extract the ZIP file containing the dataset
# ---------------------------------------------------------------
@st.cache_data
def download_and_extract():
    # URL of the ZIP file in your GitHub repository (use the raw URL)
    zip_url = 'https://github.com/waqasahmed138/FOLKsonomy/raw/main/diabetes_subset_rf2.zip'  # Update to your GitHub raw URL

    # Define the local path where the ZIP file will be stored
    zip_file_path = '/tmp/diabetes_subset_rf2.zip'  # Temporary directory in Colab/Streamlit Cloud
    extract_path = './diabetes_subset_rf2'  # Extracted folder path in the current working directory

    # Download the ZIP file
    st.write(f"🔄 Downloading ZIP file from: {zip_url}")
    r = requests.get(zip_url)
    with open(zip_file_path, 'wb') as f:
        f.write(r.content)

    # Extract the ZIP file
    extract_zip(zip_file_path, extract_path)

    # Debugging: List the files to verify the extraction
    st.write(f"📂 Extracted files in {extract_path}: {os.listdir(extract_path)}")

    return extract_path

# ---------------------------------------------------------------
# Load the Diabetes Subset RF2 files
# ---------------------------------------------------------------
def load_rf2_data():
    # Extract the data files from the ZIP
    extract_path = download_and_extract()

    # Define the paths for the `.tsv` files relative to the extracted folder
    desc_path = f"{extract_path}/descriptions_diabetes.tsv"
    concept_path = f"{extract_path}/concepts_diabetes.tsv"
    rels_path = f"{extract_path}/relationships_diabetes.tsv"

    # Ensure files exist in the extracted directory
    if not os.path.exists(desc_path):
        st.error(f"❌ File not found: {desc_path}")
    if not os.path.exists(concept_path):
        st.error(f"❌ File not found: {concept_path}")
    if not os.path.exists(rels_path):
        st.error(f"❌ File not found: {rels_path}")

    # Read the TSV files into DataFrames
    descriptions = pd.read_csv(desc_path, sep="\t", dtype=str)
    concepts = pd.read_csv(concept_path, sep="\t", dtype=str)
    relationships = pd.read_csv(rels_path, sep="\t", dtype=str)

    return descriptions, concepts, relationships

# ---------------------------------------------------------------
# Load the RF2 dataset
# ---------------------------------------------------------------
descriptions, concepts, relationships = load_rf2_data()

# Keep active English terms only
descriptions = descriptions[descriptions["active"] == "1"]
all_terms = descriptions["term"].dropna().unique().tolist()
concept_ids = descriptions["conceptId"].tolist()

# ---------------------------------------------------------------
# Embed all terms once to calculate embeddings
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Helper: Compute similarity between user input and diabetes terms
# ---------------------------------------------------------------
def get_similarity(input_text):
    inputs = tokenizer([input_text], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    user_vec = outputs.last_hidden_state[:,0,:].cpu().numpy()
    sims = cosine_similarity(user_vec, term_embeddings)[0]
    idx = int(np.argmax(sims))
    return all_terms[idx], concept_ids[idx], float(sims[idx])

# ---------------------------------------------------------------
# Streamlit UI Layout
# ---------------------------------------------------------------
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
