# Import streamlit as st
import streamlit as st

st.set_page_config(page_title="Student Sleep Dashboard", page_icon="🌙", layout="wide")

st.Page("views/overview.py", title="Overview", icon=":material/table_chart:", default=True)
st.Page("views/explore.py", title="Explore", icon=":material/table_chart:")
st.Page("views/performance.py", title="Performance", icon=":material/table_chart:")
st.Page("views/predict.py", title="Predict", icon=":material/table_chart:")

selected_page = st.navigation([st.Page("views/overview.py", title="Overview", icon=":material/table_chart:", default=True), st.Page("views/explore.py", title="Explore", icon=":material/table_chart:"), st.Page("views/performance.py", title="Performance", icon=":material/table_chart:"), st.Page("views/predict.py", title="Predict", icon=":material/table_chart:")])

with st.sidebar.expander("About"):
    st.caption("This dashboard is designed to help students understand their sleep patterns and improve their sleep quality. It provides insights into sleep duration, sleep quality, and other related metrics.")
    st.caption("Developed by: Adam & Nabil")

selected_page.run()