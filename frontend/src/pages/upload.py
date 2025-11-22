from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import requests
import os
from io import BytesIO

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")  # backend URL
MODE = os.getenv("ENV", "dev")  # default = dev


def upload_page():

    st.title("Upload Transactions")

    # ---------------------------------------------------------
    # Compute base API endpoint depending on ENV
    # ---------------------------------------------------------
    if MODE == "dev":
        base_url = {
            "raw": f"{API_BASE}/api/predict_dev",
            "excel": f"{API_BASE}/api/predict_dev_excel",
        }
    else:
        base_url = {
            "raw": f"{API_BASE}/api/predict_prod",
            "excel": f"{API_BASE}/api/predict_prod_excel",
        }

    st.write(f"Active ENV: **{MODE}**")

    st.write("Choose an input method:")

    input_type = st.selectbox("Input Type", ["Raw Text", "Excel File"])

    # ---------------------------------------------------------
    # RAW TEXT MODE
    # ---------------------------------------------------------
    if input_type == "Raw Text":
        st.write("Enter multiple transactions (max 15):")

        raw_inputs = []
        count = st.number_input("Number of rows", min_value=1, max_value=15, value=1)

        for i in range(count):
            txt = st.text_input(f"Transaction {i+1}", key=f"tx_{i}")
            raw_inputs.append(txt)

        if st.button("Submit Raw Text"):
            payload = {"transactions": raw_inputs}

            url = base_url["raw"]

            response = requests.post(url, json=payload)
            data = response.json()

            st.session_state["results"] = data
            st.session_state["results_type"] = "table"

            st.success("Results ready! Go to Results page.")

    # ---------------------------------------------------------
    # EXCEL UPLOAD MODE
    # ---------------------------------------------------------
    else:
        st.write("Upload an Excel file")

        file = st.file_uploader("Upload .xlsx", type=["xlsx"])

        if file and st.button("Submit Excel"):
            df = pd.read_excel(file)

            buffer = BytesIO()
            df.to_excel(buffer, index=False)
            buffer.seek(0)

            url = base_url["excel"]

            response = requests.post(
                url,
                files={"file": ("uploaded.xlsx", buffer, "application/vnd.ms-excel")},
            )

            # backend responds with either JSON table or Excel file
            if response.headers.get("content-type") == "application/json":
                st.session_state["results"] = response.json()
                st.session_state["results_type"] = "table"

            else:
                st.session_state["results_file"] = response.content
                st.session_state["results_type"] = "excel"

            st.success("Results ready! Go to Results page.")
