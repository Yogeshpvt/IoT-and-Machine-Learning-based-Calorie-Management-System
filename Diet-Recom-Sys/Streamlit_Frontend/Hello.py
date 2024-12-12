import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Diet Recommendation System! 👋")

st.write("Select an option on the left")

st.write("A Dashboard created by Yogesh V, Vishaal V, Shylendra S")

streamlit_app_food_detect="http://localhost:8502"
st.sidebar.markdown(f"[Food Detection]({streamlit_app_food_detect})")

