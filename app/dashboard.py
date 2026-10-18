# Import streamlit as st
# Import sys and Path from pathlib to manipulate the system path for module imports.
import sys
from pathlib import Path
import streamlit as st

# Let the pages import from src/ (e.g. src.data_processor) by adding the project root to Python's search path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set the Streamlit page configuration with a title, icon, and layout.
st.set_page_config(page_title="Student Sleep Dashboard", page_icon="🌙", layout="wide")

# Define the navigation pages for the Streamlit app, including Overview, Explore, Performance, and Predict. Each page is associated with a specific view file and has a title and icon.
# The Overview page is set as the default page that loads when the app starts.
st.Page("views/overview.py", title="Overview", icon=":material/table_chart:", default=True)
st.Page("views/explore.py", title="Explore", icon=":material/table_chart:")
st.Page("views/performance.py", title="Performance", icon=":material/table_chart:")
st.Page("views/predict.py", title="Predict", icon=":material/table_chart:")

# Set the selected page based on user navigation and run the corresponding view file. The selected page is determined by the user's interaction with the navigation menu, and the appropriate view file is executed to display the content for that page.
selected_page = st.navigation([st.Page("views/overview.py", title="Overview", icon=":material/table_chart:", default=True), st.Page("views/explore.py", title="Explore", icon=":material/table_chart:"), st.Page("views/performance.py", title="Performance", icon=":material/table_chart:"), st.Page("views/predict.py", title="Predict", icon=":material/table_chart:")])

# Add an "About" section in the sidebar with an expander that provides information about the purpose of the dashboard and the developers. This section helps users understand the context and goals of the application, as well as acknowledge the contributions of the developers.
with st.sidebar.expander("About"):
    # Add a caption to the "About" section that describes the purpose of the dashboard, which is to help students understand their sleep patterns and improve their sleep quality. The caption also mentions that the dashboard provides insights into sleep duration, sleep quality, and other related metrics.
    st.caption("This dashboard is designed to help students understand their sleep patterns and improve their sleep quality. It provides insights into sleep duration, sleep quality, and other related metrics.")
    # Add a caption to acknowledge the developers of the dashboard, Adam and Nabil. This provides recognition for their work in creating the application and its features.
    st.caption("Developed by: Adam & Nabil")

# Run the selected page's view file to display the content for that page. The view file contains the specific components, visualizations, and interactions for the selected page, allowing users to explore different aspects of the student sleep data and predictions.
selected_page.run()