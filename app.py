import streamlit as st
import os


def get_file_content_as_string(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except PermissionError:
        return f"Error: No permission to read file at {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


st.set_page_config(
    page_title="DOC GPT",
    page_icon="📄",
    layout="wide",
)


def main():
    st.title("📄 Document GPT")

    with st.sidebar:
        st.write("Options")
        app_mode = st.selectbox("Choose the app mode",
                                ["Home", "Source code"])

    if app_mode == "Home":
        st.markdown("""
            ## Hi there🍀
                사용 방법
                \n\n
                ### blablabla
                # \n\n
            ### blablabla
            """)
    elif app_mode == "Source code":  # Changed to match the selectbox option
        st.markdown("## Source CODE")
        file_path = os.path.join("pages", "Document_gpt.py")
        content = get_file_content_as_string(file_path)
        st.code(content)

        # If there's an error, it will be displayed as text
        if content.startswith("Error"):
            st.error(content)


if __name__ == "__main__":
    main()
