import streamlit as st
from dotenv import load_dotenv
load_dotenv('.streamlit/secrets.toml')
st.secrets["chroma"]["chroma_path"]
