import streamlit as st

st.set_page_config(
    page_title="DOC GPT",
    page_icon="📄",
    layout="wide",)

st.title("📄 DOC GPT")

st.markdown("""
            # Hi there🍀
            
            ## Welcome to GPT Word
            
            ### - ✅ Document GPT(/docsgpt)
            
            ### - ✅ Code Page(/code)
            """)

with st.sidebar:
    test = st.textinput("test ...")
