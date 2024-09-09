## 목차
1. [Ullala GPT](#Ullala_GPT)
2. [첨부 01 :Document GPT](#첨부자료-01--document_gptpy-코드-설명)
3. [첨부 02 : Quiz GPT 코드](#첨부-02--quiz-gpt-코드)

# Ullala GPT

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
   - WIKI 선택 시 키워드 검색을 통해 키워드 관련 퀴즈 제공(난이도 선택 가능)
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

# 첨부 02 : Quiz GPT 코드

## 목차
1. [개요](#개요)
2. [환경 설정 및 라이브러리](#환경-설정-및-라이브러리)
3. [유틸리티 클래스 및 함수](#유틸리티-클래스-및-함수)
4. [OpenAI 모델 구성](#openai-모델-구성)
5. [문서 처리 기능](#문서-처리-기능)
6. [위키피디아 통합](#위키피디아-통합)
7. [퀴즈 생성 엔진](#퀴즈-생성-엔진)
8. [프롬프트 엔지니어링](#프롬프트-엔지니어링)
9. [사용자 인터페이스 설계](#사용자-인터페이스-설계)
10. [메인 애플리케이션 로직](#메인-애플리케이션-로직)
11. [오류 처리 및 로깅](#오류-처리-및-로깅)
12. [성능 최적화](#성능-최적화)
13. [결론 및 향후 개선 사항](#결론-및-향후-개선-사항)

## 1. 개요

Quiz GPT는 Streamlit을 기반으로 한 대화형 웹 애플리케이션으로, OpenAI의 GPT-4 모델을 활용하여 동적으로 퀴즈를 생성합니다. 사용자는 위키피디아 주제를 검색하거나 직접 파일을 업로드하여 다양한 주제에 대한 퀴즈를 만들 수 있습니다. 이 애플리케이션은 교육 목적 및 엔터테인먼트용으로 설계되었으며, 사용자 지정 난이도 설정을 지원합니다.

## 2. 환경 설정 및 라이브러리

```python
import json
import os
import requests
import logging
import wikipedia
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.schema import BaseOutputParser, Document

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
```

이 섹션에서는 애플리케이션에 필요한 모든 라이브러리를 임포트합니다:

- `json`: JSON 데이터 처리를 위해 사용됩니다.
- `os`: 운영 체제 관련 기능(예: 파일 경로 조작)에 사용됩니다.
- `requests`: HTTP 요청을 보내기 위해 사용됩니다(API 키 검증 등).
- `logging`: 애플리케이션 로깅을 위해 사용됩니다.
- `wikipedia`: 위키피디아 API와 상호 작용하기 위해 사용됩니다.
- LangChain 관련 임포트:
  - `UnstructuredFileLoader`: 다양한 형식의 파일을 로드하기 위해 사용됩니다.
  - `CharacterTextSplitter`: 긴 텍스트를 관리 가능한 청크로 분할하기 위해 사용됩니다.
  - `ChatOpenAI`: OpenAI의 채팅 모델과 상호 작용하기 위해 사용됩니다.
  - `ChatPromptTemplate`: 구조화된 프롬프트를 생성하기 위해 사용됩니다.
  - `StreamingStdOutCallbackHandler`: 스트리밍 응답을 처리하기 위해 사용됩니다.
- `streamlit`: 웹 인터페이스를 구축하기 위해 사용됩니다.

로깅 설정은 INFO 레벨로 구성되어 있어, 중요한 이벤트와 오류를 콘솔에 출력합니다.

## 3. 유틸리티 클래스 및 함수

### JsonOutputParser 클래스

```python
class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()
```

이 클래스는 LangChain의 `BaseOutputParser`를 상속받아 구현되었습니다:
- `parse` 메서드는 AI 모델의 출력을 정제하고 JSON으로 파싱합니다.
- 코드 블록 구분자(`"""`)와 'json' 키워드를 제거하여 순수한 JSON 문자열만 남깁니다.
- `json.loads()`를 사용하여 문자열을 Python 딕셔너리로 변환합니다.
- 이 파서는 AI 모델의 구조화된 출력을 애플리케이션에서 쉽게 사용할 수 있는 형태로 변환하는 데 중요한 역할을 합니다.

### API 키 검증 함수

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

이 함수는 사용자가 입력한 OpenAI API 키의 유효성을 검사합니다:
- OpenAI API의 '/v1/models' 엔드포인트에 GET 요청을 보냅니다.
- 유효한 API 키의 경우 200 상태 코드를 반환합니다.
- 네트워크 오류나 잘못된 API 키의 경우 False를 반환합니다.
- 이 검증 과정은 사용자 경험을 향상시키고, 잘못된 API 키로 인한 오류를 방지합니다.

## 4. OpenAI 모델 구성

```python
@st.cache_resource
def get_openai_model(api_key):
    return ChatOpenAI(
        temperature=0.1,
        model="gpt-4-0613",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_key=api_key,
    )
```

이 함수는 OpenAI의 GPT-4 모델을 초기화하고 구성합니다:
- `@st.cache_resource` 데코레이터를 사용하여 모델 인스턴스를 캐시합니다. 이는 반복적인 모델 초기화를 방지하여 성능을 향상시킵니다.
- `temperature=0.1`: 낮은 temperature 값을 설정하여 모델의 출력을 더 결정적이고 일관되게 만듭니다.
- `model="gpt-4-0613"`: GPT-4 모델의 특정 버전을 지정합니다.
- `streaming=True`: 스트리밍 모드를 활성화하여 실시간으로 응답을 받습니다.
- `callbacks=[StreamingStdOutCallbackHandler()]`: 스트리밍 응답을 표준 출력으로 처리합니다.
- 사용자의 API 키를 모델 구성에 포함시킵니다.

## 5. 문서 처리 기능

```python
@st.cache_data(show_spinner="파일 로딩 중...")
def split_file(file):
    file_content = file.read()
    cache_dir = "./.cache/quiz_files"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        logging.info(f"디렉토리 생성: {cache_dir}")
    file_path = os.path.join(cache_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file_content)
    logging.info(f"파일 저장: {file_path}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs
```

이 함수는 업로드된 파일을 처리하고 분할합니다:
- `@st.cache_data` 데코레이터를 사용하여 처리 결과를 캐시합니다. 이는 동일한 파일에 대한 중복 처리를 방지합니다.
- 업로드된 파일을 로컬 캐시 디렉토리에 저장합니다.
- `CharacterTextSplitter`를 사용하여 문서를 관리 가능한 청크로 분할합니다:
  - `chunk_size=600`: 각 청크의 최대 크기를 600자로 설정합니다.
  - `chunk_overlap=100`: 청크 간 100자 오버랩을 허용하여 컨텍스트 유지를 보장합니다.
- `UnstructuredFileLoader`를 사용하여 다양한 파일 형식(.docx, .txt, .pdf, .md)을 지원합니다.
- 분할된 문서 청크를 반환합니다.

## 6. 위키피디아 통합

```python
@st.cache_data(show_spinner="위키피디아 검색 중...")
def wiki_search(term):
    try:
        try:
            page = wikipedia.page(term)
            return [{"page_content": page.content, "metadata": {"title": page.title}}]
        except wikipedia.exceptions.DisambiguationError as e:
            page = wikipedia.page(e.options[0])
            return [{"page_content": page.content, "metadata": {"title": page.title}}]
        except wikipedia.exceptions.PageError:
            search_results = wikipedia.search(term, results=1)
            if not search_results:
                return []
            page = wikipedia.page(search_results[0])
            return [{"page_content": page.content, "metadata": {"title": page.title}}]
    except Exception as e:
        st.error(f"위키피디아 검색 중 오류 발생: {str(e)}")
        return []
```

이 함수는 위키피디아 API를 사용하여 주제를 검색합니다:
- `@st.cache_data` 데코레이터를 사용하여 검색 결과를 캐시합니다. 이는 동일한 검색어에 대한 반복적인 API 호출을 방지합니다.
- 검색 프로세스는 여러 단계로 구성됩니다:
  1. 정확한 제목 매치 시도
  2. 모호성 해결 (DisambiguationError 처리)
  3. 검색 결과가 없을 경우 유사한 페이지 검색
- 각 단계에서 예외를 처리하여 견고성을 보장합니다.
- 검색 결과를 표준화된 형식 (페이지 내용과 메타데이터)으로 반환합니다.

## 7. 퀴즈 생성 엔진

```python
@st.cache_data(show_spinner="퀴즈 만드는 중...")
def run_quiz_chain(_docs, topic, _llm, difficulty):
    if not isinstance(_docs, list) or len(_docs) == 0:
        return {"questions": []}

    formatted_docs = format_docs(_docs)

    questions_chain = {
        "context": lambda x: formatted_docs,
        "difficulty": lambda x: difficulty
    } | questions_prompt | _llm
    formatting_chain = formatting_prompt | _llm
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(formatted_docs)
```

이 함수는 LangChain을 사용하여 퀴즈 생성 프로세스를 구현합니다:
- `@st.cache_data` 데코레이터를 사용하여 동일한 입력에 대한 퀴즈 생성 결과를 캐시합니다.
- 입력 검증을 수행하여 빈 문서 리스트에 대해 빈 퀴즈를 반환합니다.
- `format_docs` 함수(별도 정의 필요)를 사용하여 문서를 표준 형식으로 변환합니다.
- LangChain의 체인 개념을 사용하여 퀴즈 생성 프로세스를 구성합니다:
  1. `questions_chain`: 문맥과 난이도를 고려하여 질문을 생성합니다.
  2. `formatting_chain`: 생성된 질문을 구조화된 형식으로 변환합니다.
  3. `output_parser`: 최종 출력을 파싱하여 사용