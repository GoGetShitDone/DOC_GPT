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


def display_code_tab(title, file_name):
    st.markdown(f"##{title}")
    file_path = os.path.join("pages", file_name)
    content = get_file_content_as_string(file_path)
    st.code(content)
    if content.startswith("Error"):
        st.error(content)


st.set_page_config(
    page_title="DOC GPT",
    page_icon="📄",
    layout="wide",
)


def main():
    st.title("🐻 Ullala GPT")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🐝 README", "🐝 DOCUMENT GPT CODE", "🐝 QUIZ GPT CODE", "🐝 SITE GPT CODE"])

    with tab1:
        st.markdown("""
                        ### 🍯 README
                        업로드한 문서를 기반으로 AI에게 질문하고 답변을 받을 수 있으며, 자료 기반 퀴즈 생성이 가능합니다.
                            \n
                            1. app 페이지
                                - 기본 안내사항을 받을 수 있습니다. 
                                - 페이지 내 탭을 통해 Source Code를 확인할 수 있습니다. 
                            \n
                            2. Document 페이지
                                - API 키를 입력하고 유효성 검증합니다. 
                                - 문서를 첨부하고 문서를 바탕으로 AI에게 질문하고 답변받을 수 있습니다. 
                                - 첨부 문서는 .txt, .pdf, .docx, .md 파일을 지원합니다. 
                            \n
                            3. Quiz GPT 페이지
                                - OpenAI API 키 입력 및 유효성 검증
                                - WIKI 또는 파일 업로드 선택 가능 
                                - WIKI 선택 시 키워드 검색을 통해 키워드 관련 퀴즈 제공 
                                - 파일 업로드 선택 시 업로드 문서(.txt, .pdf, .docx, .md 파일 지원)에 따른 퀴즈 제공
                            \n
                            4. Site GPT 페이지
                                - OpenAI API 키 입력 및 유효성 검증
                                - Web site 주소 입력 후 관련 검색 가능 
                                - .xml 형태의 Web site 주소 입력 필요
                            \n
                            4. 파일 구조 
                                .
                                ├── .gitignore
                                ├── app.py
                                ├── pages
                                │   └── Document_gpt.py
                                │   ├── Quiz_gpt.py
                                │   └── Site_gpt.py
                                └── requirements.txt
                                
                        """)

    with tab2:
        display_code_tab("## 🍯 Document GPT CODE", "Document_gpt.py")

    with tab3:
        display_code_tab("## 🍯 Quiz GPT CODE", "Quiz_gpt.py")

    with tab4:
        display_code_tab("## 🍯 Site GPT CODE", "site_gpt.py")

    with st.sidebar:
        st.markdown('<a href="https://github.com/GoGetShitDone/DOC_GPT" target="_blank"><button style="background-color:#0F1116;color:white;padding:10px 30px;border:none;border-radius:5px;cursor:pointer;">🍯 Ullala GitHub</button></a>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
