import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import altair as alt
from utility import connect_gsheet

# --- GOOGLE SHEETS CONNECTION ---
# This function is cached to prevent re-running on every page reload.


def app():
    st.header("Admit Card")
    try:
        # Step 1: connect client
        client = connect_gsheet()
        
        # Check if connection failed (returns None instead of calling st.stop())
        if client is None:
            st.warning("⚠️ Google Sheets connection is not available. This module requires Google Sheets access.")
            st.info("💡 Please configure Google Sheets credentials in secrets.toml or use a different module.")
            return

        # Step 2: open sheet by ID
        if "my_secrets" not in st.secrets or "sheet_id" not in st.secrets["my_secrets"]:
            st.error("Sheet ID not found in secrets. Please configure secrets.toml.")
            return
            
        sh = client.open_by_key(st.secrets["my_secrets"]["sheet_id"])

     

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return

        
