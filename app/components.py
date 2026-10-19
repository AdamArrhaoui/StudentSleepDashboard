"""
Reusable building blocks shared by all dashboard pages.

"""
import streamlit as st

# Page_header function: displays a title, description, and divider at the top of every page. Added a flag to switch to markdown to ensure the description is displayed in a larger font size and gray color, as per the design requirements can be displayed in a larger size. 
def page_header(title: str, description: str, isCaption: bool = True) -> None:
    """Standard header at the top of every page: title, one-line purpose, divider."""
    st.title(title)
    if isCaption:
        st.caption(description)
    elif isCaption is False:
        st.markdown( f"<p style='font-size:20px; color:gray;'>{description}</p>",
    unsafe_allow_html=True)
    st.divider()

# Regular section header function: displays a subheader and divider for each section within a page. This function is used to create a clear separation between different sections of content, making it easier for users to navigate and understand the information presented on the page.
def section_header(text: str) -> None:
    """Heading for a section inside a page, with the theme's teal divider."""
    st.subheader(text, divider="blue")
