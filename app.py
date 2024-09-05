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
            ## 🍀 README 🍀
            Document GPT는 대화형 문서 분석 애플리케이션입니다. 업로드한 문서를 기반으로 AI에게 질문하고 답변을 받을 수 있습니다.
                \n
                1. app 페이지
                    - 기본 안내사항을 받을 수 있습니다. 
                    - 옵션에서 Source Code 를 선택해서 코드를 확인할 수 있습니다. 
                \n
                2. Document 페이지
                    - API 키를 입력하고 유효성 검증합니다. 
                    - 문서를 첨부하고 문서를 바탕으로 AI에게 질문하고 답변받을 수 있습니다. 
                    - 첨부 문서는 .txt, .pdf, .docx, .md 파일을 지원합니다. 
                \n
                3. 기타: 파일 구조 
                    .
                    ├── .gitignore
                    ├── app.py
                    ├── pages
                    │   └── Document_gpt.py
                    └── requirements.txt
                    # Document GPT
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
