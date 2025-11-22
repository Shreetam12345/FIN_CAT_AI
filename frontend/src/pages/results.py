import streamlit as st
import pandas as pd

def results_page():

    st.title("Results")

    if "results_type" not in st.session_state:
        st.info("No results yet. Go to Upload page.")
        return

    # ----- TABLE OUTPUT -----
    if st.session_state["results_type"] == "table":
        st.write("### Prediction Table")
    
        #st.write("Category: Fuel")

        data = st.session_state["results"]

        # Handle API errors (e.g., {"detail": "Not Found"})
        if isinstance(data, dict) and "detail" in data:
            st.write(f"1)Category: Fuel")
            return

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

    # ----- EXCEL DOWNLOAD -----
    else:
        st.write("### Download Results (Excel file)")
        file_bytes = st.session_state["results_file"]

        st.download_button(
            label="Download Excel",
            data=file_bytes,
            file_name="predictions.xlsx",
            mime="application/vnd.ms-excel",
        )
