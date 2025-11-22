import streamlit as st
import os
import requests

st.set_page_config(page_title="AI Transaction Classifier", layout="wide")

# ---------------------------------------------------
# FETCH ENV FROM BACKEND (BECAUSE FRONTEND HAS NO .env)
# ---------------------------------------------------
if "ENV" not in st.session_state:
    try:
        r = requests.get("http://localhost:8000/api/env")
        st.session_state.ENV = r.json().get("env", "dev")
    except:
        st.session_state.ENV = "dev"

ENV = st.session_state.ENV

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload", "Results"])
st.sidebar.markdown(f"**ENV:** `{ENV}`")

# Load pages
if page == "Upload":
    from src.pages.upload import upload_page
    upload_page()

elif page == "Results":
    from src.pages.results import results_page
    results_page()
