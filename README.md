# DOC GPT

Document GPT는 Streamlit을 이용해 만든 대화형 문서 분석 애플리케이션입니다. 사용자가 업로드한 문서를 기반으로 AI에게 질문하고 답변을 받을 수 있습니다.

## 기능

1. **홈 페이지(app)**
   - 애플리케이션에 대한 기본 안내 제공
   - 각각의 Tab에서 관련 소스 코드 전체를 확인 가능

2. **Document GPT 페이지**
   - OpenAI API 키 입력 및 유효성 검증
   - 문서 업로드 (.txt, .pdf, .docx, .md 파일 지원)
   - 업로드된 문서에 대해 AI에게 질문하고 답변 받기

3. **Quiz GPT 페이지**
   - OpenAI API 키 입력 및 유효성 검증
   - WIKI 또는 파일 업로드 선택 가능 
   - WIKI 선택 시 키워드 검색을 통해 키워드 관련 퀴즈 제공 
   - 파일 업로드 선택 시 업로드 문서(.txt, .pdf, .docx, .md 파일 지원)에 따른 퀴즈 제공

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
│   ├── Document_gpt.py
│   └── Quiz_gpt.py
└── requirements.txt
```

- `app.py`: 메인 Streamlit 애플리케이션 파일
- `pages/Document_gpt.py`: Document GPT 기능을 구현한 페이지
- `pages/Quiz_gpt.py`: 첨부문서 또는 위키의 키워드 관련 퀴즈를 제공하는 페이지
- `requirements.txt`: 프로젝트 의존성 목록

## 사용 방법

1. OpenAI API 키를 사이드바에 입력합니다.
2. 분석하고자 하는 문서 파일을 업로드합니다.
3. 문서에 대해 질문을 입력하면 AI가 답변을 제공합니다.

## 주의사항

- OpenAI API 키는 애플리케이션에 저장되지 않습니다.
- 지원되는 파일 형식: .txt, .pdf, .docx, .md

---
---

# 첨부자료 01 : Document_gpt.py 코드 설명

## 목차
1. [개요](#개요)
2. [라이브러리 임포트](#라이브러리-임포트)
3. [로깅 설정](#로깅-설정)
4. [Streamlit 페이지 설정](#streamlit-페이지-설정)
5. [ChatCallbackHandler 클래스](#chatcallbackhandler-클래스)
6. [OpenAI 모델 초기화](#openai-모델-초기화)
7. [파일 임베딩](#파일-임베딩)
8. [메시지 관리 함수](#메시지-관리-함수)
9. [문서 포맷팅](#문서-포맷팅)
10. [프롬프트 템플릿](#프롬프트-템플릿)
11. [API 키 검증](#api-키-검증)
12. [메인 UI 구성](#메인-ui-구성)
13. [메인 로직](#메인-로직)
14. [결론](#결론)

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
    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        self.message = ""  # Reset the message after saving

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        if self.message_box:
            self.message_box.markdown(self.message)
```

OpenAI의 스트리밍 응답을 실시간으로 화면에 표시하기 위한 콜백 핸들러입니다. 토큰이 생성될 때마다 메시지를 업데이트하고 표시합니다.

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

OpenAI 모델을 초기화하고 캐싱합니다. 스트리밍 모드를 사용하여 실시간 응답을 가능하게 합니다.

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

    file_content = file.read()
    with open(file_path, "wb") as f:
        f.write(file_content)

    embeddings_store = LocalFileStore(os.path.join(embeddings_dir, file.name))
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, embeddings_store)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever
```

업로드된 파일을 처리하고 임베딩합니다. 캐시 디렉토리를 생성하고 관리합니다. 파일을 청크로 분할하고, 임베딩을 생성한 후 FAISS 벡터 저장소에 저장합니다.

## 메시지 관리 함수

```python
def save_message(message, role):
    # Remove 'content=' prefix if it exists
    if message.startswith("content='") and message.endswith("'"):
        # Remove the first 9 characters and the last character
        message = message[9:-1]
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        if role == "ai":
            st.empty().markdown(message)
        else:
            st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)
```

채팅 히스토리를 관리하는 함수들입니다. `save_message`는 메시지를 세션 상태에 저장하고, `send_message`는 메시지를 화면에 표시합니다. `paint_history`는 저장된 모든 메시지를 화면에 다시 그립니다.

## 문서 포맷팅

```python
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)
```

검색된 문서들을 하나의 문자열로 포맷팅합니다.

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

AI에게 전달할 프롬프트의 구조를 정의합니다. 시스템 메시지와 사용자 질문으로 구성됩니다.

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

제공된 OpenAI API 키의 유효성을 검사합니다. OpenAI의 모델 리스트 엔드포인트를 호출하여 확인합니다.

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

애플리케이션의 메인 UI를 구성합니다. 제목, 설명, API 키 입력 필드, 파일 업로더를 포함합니다.

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
                        response = chain.invoke(message)
                        save_message(response.content, "ai")
            except Exception as e:
                st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                logging.error(
                    f"Error processing file: {str(e)}", exc_info=True)
        else:
            st.warning("Please upload a file in the sidebar.")
    else:
        st.error("Invalid API key. Please check your OpenAI API key and try again.")
elif not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
```

애플리케이션의 주요 로직을 구현합니다:
1. API 키 유효성 검사
2. 파일 업로드 및 처리
3. 사용자 입력 처리
4. AI 응답 생성 및 표시
5. 오류 처리 및 로깅

메시지 히스토리를 세션 상태에 저장하여 대화의 연속성을 유지합니다.

## 결론

이 Document GPT 애플리케이션은 Streamlit과 LangChain을 활용하여 복잡한 AI 기능을 간단한 웹 인터페이스로 구현했습니다. 주요 특징으로는 실시간 응답, 문서 기반 질문답변, API 키 보안, 효율적인 파일 처리, 그리고 향상된 오류 처리 및 로깅이 있습니다. 이 애플리케이션은 사용자 친화적인 인터페이스와 강력한 AI 기능을 결합하여 문서 분석 및 정보 추출 작업을 효과적으로 수행할 수 있게 해줍니다.

---
---

# 첨부자료 02 : Quiz_gpt.py 코드 설명

## 목차
1. [개요](#개요)
2. [라이브러리 임포트](#라이브러리-임포트)
3. [로깅 설정](#로깅-설정)
4. [JsonOutputParser 클래스](#jsonoutputparser-클래스)
5. [Streamlit 페이지 설정](#streamlit-페이지-설정)
6. [API 키 검증](#api-키-검증)
7. [OpenAI 모델 초기화](#openai-모델-초기화)
8. [문서 포맷팅](#문서-포맷팅)
9. [퀴즈 생성 체인](#퀴즈-생성-체인)
10. [위키피디아 검색](#위키피디아-검색)
11. [파일 분할 및 로딩](#파일-분할-및-로딩)
12. [프롬프트 템플릿](#프롬프트-템플릿)
13. [메인 UI 구성](#메인-ui-구성)
14. [메인 로직](#메인-로직)
15. [결론](#결론)

## 개요

이 코드는 Streamlit을 사용하여 만든 "Quiz GPT" 웹 애플리케이션입니다. 사용자가 위키피디아 주제를 검색하거나 파일을 업로드하면, OpenAI의 API를 활용하여 해당 내용에 대한 퀴즈를 생성합니다.

## 라이브러리 임포트

```python
import json
import os
import requests
import logging
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser, Document
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

## JsonOutputParser 클래스

```python
class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()
```

AI 모델의 출력을 JSON 형식으로 파싱하는 클래스입니다. 코드 블록 표시와 'json' 키워드를 제거한 후 JSON으로 파싱합니다.

## Streamlit 페이지 설정

```python
st.set_page_config(
    page_title="Quiz GPT",
    page_icon="🕹️",
    layout="wide",
)

st.title("🕹️ Quiz GPT")
```

Streamlit 애플리케이션의 기본 설정과 제목을 정의합니다.

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

제공된 OpenAI API 키의 유효성을 검사합니다. OpenAI의 모델 리스트 엔드포인트를 호출하여 확인합니다.

## OpenAI 모델 초기화

```python
@st.cache_resource
def get_openai_model(api_key):
    return ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_key=api_key,
    )
```

OpenAI 모델을 초기화하고 캐싱합니다. 스트리밍 모드를 사용하여 실시간 응답을 가능하게 합니다.

## 문서 포맷팅

```python
def format_docs(docs):
    if isinstance(docs, list):
        if all(isinstance(doc, dict) and "page_content" in doc for doc in docs):
            return "\n\n".join(doc["page_content"] for doc in docs)
        elif all(isinstance(doc, str) for doc in docs):
            return "\n\n".join(docs)
        elif all(hasattr(doc, 'page_content') for doc in docs):
            return "\n\n".join(doc.page_content for doc in docs)
    elif isinstance(docs, str):
        return docs
    else:
        raise ValueError("Unsupported document format")
```

다양한 형식의 문서를 일관된 문자열 형식으로 변환합니다.

## 퀴즈 생성 체인

```python
@st.cache_data(show_spinner="Making Quiz...")
def run_quiz_chain(_docs, topic, _llm):
    if not isinstance(_docs, list) or len(_docs) == 0:
        return {"questions": []}

    formatted_docs = format_docs(_docs)

    questions_chain = {
        "context": lambda x: formatted_docs} | questions_prompt | _llm
    formatting_chain = formatting_prompt | _llm
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(formatted_docs)
```

문서 내용을 바탕으로 퀴즈를 생성하는 함수입니다. LangChain의 체인 개념을 사용하여 질문 생성과 포맷팅을 순차적으로 수행합니다.

## 위키피디아 검색

```python
@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=1, lang="en")
    docs = retriever.get_relevant_documents(term)

    if docs and isinstance(docs[0], Document):
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    return docs
```

위키피디아에서 주어진 주제에 대한 정보를 검색하고 반환합니다.

## 파일 분할 및 로딩

```python
@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    cache_dir = "./.cache/quiz_files"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        logging.info(f"Created directory: {cache_dir}")
    file_path = os.path.join(cache_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file_content)
    logging.info(f"File saved: {file_path}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs
```

업로드된 파일을 읽고, 캐시에 저장한 후, 내용을 적절한 크기의 청크로 분할합니다.

## 프롬프트 템플릿

```python
questions_prompt = ChatPromptTemplate.from_messages([...])
formatting_prompt = ChatPromptTemplate.from_messages([...])
```

AI에게 전달할 프롬프트의 구조를 정의합니다. `questions_prompt`는 퀴즈 질문을 생성하기 위한 것이고, `formatting_prompt`는 생성된 질문을 JSON 형식으로 포맷팅하기 위한 것입니다.

## 메인 UI 구성

```python
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        ("Wikipedia Article", "File",),
    )
    # ... (API 키 검증 및 소스 선택 로직)
```

사이드바에 API 키 입력, 소스 선택(위키피디아 또는 파일 업로드) 옵션을 제공합니다.

## 메인 로직

```python
if not docs:
    st.markdown(
        "Welcome to Quiz GPT. Please enter your API key and choose a source in the sidebar.")
elif api_key and is_valid_api_key(api_key):
    try:
        response = run_quiz_chain(
            docs, topic if topic else file.name, _llm=llm)
        if "questions" in response and len(response["questions"]) > 0:
            with st.form("questions_form"):
                for question in response["questions"]:
                    st.write(question["question"])
                    value = st.radio(
                        "Select an option.", [answer["answer"]
                                              for answer in question["answers"]],
                        index=None,
                    )
                    if {"answer": value, "correct": True} in question["answers"]:
                        st.success("Correct!")
                    elif value is not None:
                        st.error("Wrong!")
                button = st.form_submit_button()
        else:
            st.error(
                "No questions were generated. Please try a different topic or source.")
    except Exception as e:
        st.error(f"An error occurred while generating the quiz: {str(e)}")
        logging.error(f"Error in run_quiz_chain: {str(e)}", exc_info=True)
```

애플리케이션의 주요 로직을 구현합니다:
1. API 키 유효성 검사
2. 문서 소스에 따른 처리 (위키피디아 검색 또는 파일 업로드)
3. 퀴즈 생성
4. 생성된 퀴즈 표시 및 사용자 응답 처리
5. 오류 처리 및 로깅

## 결론

이 Quiz GPT 애플리케이션은 Streamlit과 LangChain을 활용하여 동적인 퀴즈 생성 기능을 구현했습니다. 주요 특징으로는 위키피디아 검색 기능, 파일 업로드 지원, OpenAI API를 활용한 퀴즈 생성, 사용자 친화적인 인터페이스 등이 있습니다. 이 애플리케이션은 교육 목적이나 재미를 위한 퀴즈 생성에 효과적으로 사용될 수 있으며, 다양한 소스로부터 지식을 테스트할 수 있는 유연한 플랫폼을 제공합니다.