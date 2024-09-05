# Document GPT

Document GPT는 Streamlit을 이용해 만든 대화형 문서 분석 애플리케이션입니다. 사용자가 업로드한 문서를 기반으로 AI에게 질문하고 답변을 받을 수 있습니다.

## 기능

1. **홈 페이지**
   - 애플리케이션에 대한 기본 안내 제공
   - 소스 코드 확인 옵션

2. **Document GPT 페이지**
   - OpenAI API 키 입력 및 유효성 검증
   - 문서 업로드 (.txt, .pdf, .docx, .md 파일 지원)
   - 업로드된 문서에 대해 AI에게 질문하고 답변 받기

## 설치 및 실행

1. 저장소를 클론합니다:
   ```
   git clone [저장소 URL]
   cd [프로젝트 디렉토리]
   ```

2. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

3. Streamlit 앱을 실행합니다:
   ```
   streamlit run app.py
   ```

## 파일 구조

```
.
├── .gitignore
├── README.md
├── app.py
├── pages
│   └── Document_gpt.py
└── requirements.txt
```

- `app.py`: 메인 Streamlit 애플리케이션 파일
- `pages/Document_gpt.py`: Document GPT 기능을 구현한 페이지
- `requirements.txt`: 프로젝트 의존성 목록

## 사용 방법

1. OpenAI API 키를 사이드바에 입력합니다.
2. 분석하고자 하는 문서 파일을 업로드합니다.
3. 문서에 대해 질문을 입력하면 AI가 답변을 제공합니다.

## 주의사항

- OpenAI API 키는 애플리케이션에 저장되지 않습니다.
- 지원되는 파일 형식: .txt, .pdf, .docx, .md


# 첨부 : Document_gpt.py 코드 설명

## 목차
1. [개요](#개요)
2. [라이브러리 임포트](#라이브러리-임포트)
3. [로깅 설정](#로깅-설정)
4. [Streamlit 페이지 설정](#streamlit-페이지-설정)
5. [ChatCallbackHandler 클래스](#chatcallbackhandler-클래스)
6. [OpenAI 모델 초기화](#openai-모델-초기화)
7. [파일 임베딩](#파일-임베딩)
8. [메시지 관리 함수](#메시지-관리-함수)
9. [프롬프트 템플릿](#프롬프트-템플릿)
10. [API 키 검증](#api-키-검증)
11. [메인 UI 구성](#메인-ui-구성)
12. [메인 로직](#메인-로직)
13. [결론](#결론)

## 개요

이 코드는 Streamlit을 사용하여 만든 "Document GPT" 웹 애플리케이션입니다. 사용자가 문서를 업로드하고 해당 문서에 대해 질문할 수 있으며, OpenAI의 API를 활용하여 문서 내용을 바탕으로 답변을 제공합니다.

## 라이브러리 임포트

```python
import os
import requests
import logging
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
```

주요 라이브러리:
- Streamlit: 웹 애플리케이션 구축
- LangChain: AI 모델 활용을 위한 프레임워크
- Requests: HTTP 요청 처리
- Logging: 로그 기록

## 로깅 설정

```python
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
)
```

애플리케이션의 로그를 INFO 레벨로 설정하고, 로그 형식을 지정합니다.

## Streamlit 페이지 설정

```python
st.set_page_config(
    page_title="Document GPT",
    page_icon="📄",
    layout="wide",
)
```

Streamlit 애플리케이션의 기본 설정을 정의합니다.

## ChatCallbackHandler 클래스

```python
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
```

OpenAI의 스트리밍 응답을 실시간으로 화면에 표시하기 위한 콜백 핸들러입니다.

## OpenAI 모델 초기화

```python
@st.cache_resource
def get_openai_model(api_key):
    return ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        openai_api_key=api_key,
    )
```

OpenAI 모델을 초기화하고 캐싱합니다.

## 파일 임베딩

```python
@st.cache_data(show_spinner="Embedding File...")
def embed_file(file, api_key):
    # 캐시 디렉토리 생성
    cache_dir = "./.cache"
    files_dir = os.path.join(cache_dir, "files")
    embeddings_dir = os.path.join(cache_dir, "embeddings")

    for dir_path in [cache_dir, files_dir, embeddings_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Created directory: {dir_path}")

    file_path = os.path.join(files_dir, file.name)
    logging.info(f'file_path: {file_path}')

    # ... (파일 읽기, 텍스트 분할, 임베딩 생성 등)
    return retriever
```

업로드된 파일을 처리하고 임베딩합니다. 캐시 디렉토리를 생성하고 관리합니다.

## 메시지 관리 함수

```python
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)
```

채팅 히스토리를 관리하는 함수들입니다.

## 프롬프트 템플릿

```python
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)
```

AI에게 전달할 프롬프트의 구조를 정의합니다.

## API 키 검증

```python
def is_valid_api_key(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(
            "https://api.openai.com/v1/models", headers=headers)
        return response.status_code == 200
    except requests.RequestException:
        return False
```

제공된 OpenAI API 키의 유효성을 검사합니다.

## 메인 UI 구성

```python
st.title("📄 Document GPT")

st.markdown("환영합니다!<br>이 챗봇을 사용하여 파일을 업로드 하고 AI에게 질문하세요!<br>사이드 바에 API 키를 입력하고 파일을 업로드하세요!<br>API 키는 저장되지 않습니다.",
            unsafe_allow_html=True)

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    file = st.file_uploader("Upload a .txt, .pdf, .docs, .md files only", type=[
                            "pdf", "txt", "docx", "md"])
```

애플리케이션의 메인 UI를 구성합니다.

## 메인 로직

```python
if api_key:
    if is_valid_api_key(api_key):
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API 키가 유효합니다.")

        if file:
            try:
                retriever = embed_file(file, api_key)
                llm = get_openai_model(api_key)
                send_message("Good! Ask Anything!", "ai", save=False)
                paint_history()
                message = st.chat_input("Ask Anything! about your file...")
                if message:
                    send_message(message, "human")
                    chain = ({
                        "context": retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough(),
                    } | prompt | llm)
                    with st.chat_message("ai"):
                        chain.invoke(message)
            except Exception as e:
                st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                logging.error(f"Error processing file: {str(e)}", exc_info=True)
        else:
            st.warning("Please upload a file in the sidebar.")
    else:
        st.error("Invalid API key. Please check your OpenAI API key and try again.")
elif not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
```

애플리케이션의 주요 로직을 구현합니다:
1. API 키 유효성 검사
2. 파일 업로드 및 처리
3. 사용자 입력 처리
4. AI 응답 생성 및 표시
5. 오류 처리 및 로깅

## 결론

이 Document GPT 애플리케이션은 Streamlit과 LangChain을 활용하여 복잡한 AI 기능을 간단한 웹 인터페이스로 구현했습니다. 주요 특징으로는 실시간 응답, 문서 기반 질문답변, API 키 보안, 효율적인 파일 처리, 그리고 향상된 오류 처리 및 로깅이 있습니다. 이 애플리케이션은 사용자 친화적인 인터페이스와 강력한 AI 기능을 결합하여 문서 분석 및 정보 추출 작업을 효과적으로 수행할 수 있게 해줍니다.