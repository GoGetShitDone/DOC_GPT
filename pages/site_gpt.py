"""
1. 개요
    이번 과제는 sitemap.xlm 로드 / 파싱하는데 시간이 오래걸려서 토큰, 벡터스토어 캐시를 직접 로컬 또는 github 캐시 폴더에 저장을 해서 스트림릿으로 빠르게 업로드 될 수 있게 하려고 했나 실패함.(실패한 코드는 아래에)
    - 오랜 시간이 걸리는 근본적인 원인을 해결하지 않고 꼼수를 쓰려고 했다가 결국 아무것도 못한케이스로 남게된 과제임 
    - 지속적인 체감 난이도 증가로 따라가다가 이번 과제에서 정체됨... 
    - 아래 TA's 솔루현 해결 코드와 설명을 보고 이해할 수 있었음.

1. OpenAI API 키 입력 받기
    st.text_input을 사용하여 사용자의 OpenAI API 키를 입력받습니다.
    입력받은 API 키를 ChatOpenAI와 OpenAIEmbeddings 클래스를 사용할 때 openai_api_key 매개변수로 넘깁니다.

2. 사이트맵 로드
    SitemapLoader를 사용하여 Cloudflare의 사이트맵을 로드합니다.
    이때, 특정 URL 규칙으로 필터링하여 해당 문서만 파싱합니다. (filter_urls 이용)
    파싱된 문서에서 불필요한 부분(header와 footer)을 제거합니다. (parsing_function 이용)
    사이트맵 로드에 대한 자세한 설명은 공식 문서 (https://python.langchain.com/v0.2/docs/integrations/document_loaders/sitemap/)를 참고하세요.

3. 임베딩 및 벡터 스토어 생성
    로드된 문서의 임베딩 과정을 거치고 FAISS를 사용하여 벡터 스토어를 생성합니다.
    벡터 스토어의 as_retriever 메소드를 이용하여 retriever를 생성하고, 이것을 체인에서 사용합니다.
    (관련 공식 문서 (https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/vectorstore/))

4. 질문에 대한 답변 및 메시지 저장
    사용자가 질문을 하면 생성된 retriever와 적절한 프롬프트를 이용하여 답변을 생성하고 화면에 표시합니다.
    메시지 기록을 유지하기 위해 st.session_state를 사용하여 메시지를 저장하고 불러옵니다.
    (공식 문서(https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state))

5. 결론
    특정 사이트의 사이트맵을 로드하여 문서를 파싱하고, 이에 대한 사용자의 질문에 적절한 답변을 생성하는 챗봇을 구현하는 챌린지였습니다.
    이번 챌린지는 강의 코드를 약간만 수정하면 쉽게 제출할 수 있었습니다. 따라서 챌린지를 끝내고 바로 다음 과정으로 넘어가기보다는 코드를 분석하고 이해하는 데 집중하는 것이 중요합니다.
"""

import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.                                                  
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.

    Context: {context}
    Examples:                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5                                                 
    
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0                                                  
    
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]

    answers_chain = answers_prompt | llm_for_get_answer
    return {
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
        "question": question,
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.
            Choose the most informed answer among the answers with the same score.

            You should always respond to the source.

            Answers: {answers}
            ---
            Examples:

            The moon is 384,400 km away.

            Source: https://example.com
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]

    choose_chain = choose_prompt | llm_for_choose_answer
    condensed = "\n\n".join(
        f"{answer['answer']} \nSource:{answer['source']} \nDate:{answer['date']} \n\n"
        for answer in answers
    )

    return choose_chain.invoke({"answers": condensed, "question": question})


def parse_page(soup: BeautifulSoup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", "")


@st.cache_data(show_spinner="Loading Website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )

    loader = SitemapLoader(
        url,
        # filter_urls=[],
        filter_urls=(
            [
                r"https:\/\/developers.cloudflare.com/ai-gateway.*",
                r"https:\/\/developers.cloudflare.com/vectorize.*",
                r"https:\/\/developers.cloudflare.com/workers-ai.*",
            ]
        ),
        parsing_function=parse_page,
    )
    # loader.requests_per_second = 1
    ua = UserAgent()
    loader.headers = {"User-Agent": ua.random}
    docs = loader.load_and_split(splitter)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    url_copy = url[:]
    cache_filename = url_copy.replace("/", "_")
    cache_filename.strip()
    cache_dir = LocalFileStore(f"./.cache/{cache_filename}/")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever()


# Chat & Streaming
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


if "messages" not in st.session_state:
    st.session_state["messages"] = []


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


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
    layout="wide",
)

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    st.markdown('<a href="https://github.com/GoGetShitDone/DOC_GPT" target="_blank"><button style="background-color:#0F1116;color:white;padding:10px 30px;border:none;border-radius:5px;cursor:pointer;">🍯 Ullala GitHub</button></a>', unsafe_allow_html=True)
    openai_api_key = st.text_input("Input your OpenAI API Key")
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
        value="https://developers.cloudflare.com/sitemap.xml",
        disabled=True,
    )
if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL(.xml)")
    if not openai_api_key:
        st.error("Please input your OpenAI API Key on the sidebar")
    else:
        paint_history()
        llm_for_get_answer = ChatOpenAI(
            temperature=0.1,
            openai_api_key=openai_api_key,
        )
        llm_for_choose_answer = ChatOpenAI(
            temperature=0.1,
            openai_api_key=openai_api_key,
            streaming=True,
            callbacks=[ChatCallbackHandler()],
        )

        retriever = load_website(url)
        query = st.chat_input("Ask a question to the website.")
        if query:
            send_message(query, "human")
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            with st.chat_message("ai"):
                chain.invoke(query)


# --------------------------------------
# 기존 시도했던 코드

# import streamlit as st
# import requests
# import os
# import pickle
# import hashlib
# import time
# import traceback
# import numpy as np
# from langchain.document_loaders import SitemapLoader
# from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.faiss import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
# from openai.error import RateLimitError
# from langchain.storage import LocalFileStore
# from langchain.embeddings import CacheBackedEmbeddings

# st.set_page_config(
#     page_title="Site GPT",
#     page_icon="🌐",
#     layout="wide",
# )

# # Initialize session state
# if 'api_key' not in st.session_state:
#     st.session_state.api_key = ""
# if 'llm' not in st.session_state:
#     st.session_state.llm = None
# if 'retriever' not in st.session_state:
#     st.session_state.retriever = None

# CACHE_DIR = os.path.join(os.path.dirname(
#     os.path.dirname(__file__)), '.cache', 'site_files')
# os.makedirs(CACHE_DIR, exist_ok=True)


# def get_cache_path(url):
#     return os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + '.pkl')


# def save_to_cache(url, data):
#     with open(get_cache_path(url), 'wb') as f:
#         pickle.dump(data, f)


# def load_from_cache(url, max_age=604800):
#     cache_path = get_cache_path(url)
#     if os.path.exists(cache_path):
#         mod_time = os.path.getmtime(cache_path)
#         if time.time() - mod_time < max_age:
#             with open(cache_path, 'rb') as f:
#                 return pickle.load(f)
#     return None


# @st.cache_data(show_spinner=False)
# def is_valid_api_key(api_key):
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }
#     try:
#         response = requests.get(
#             "https://api.openai.com/v1/models", headers=headers)
#         return response.status_code == 200
#     except requests.RequestException:
#         return False


# @st.cache_resource
# def get_llm(api_key):
#     return ChatOpenAI(temperature=0.1, openai_api_key=api_key)


# answers_prompt = ChatPromptTemplate.from_template(
#     """
#     Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

#     Then, give a score to the answer between 0 and 5.
#     If the answer answers the user question the score should be high, else it should be low.
#     Make sure to always include the answer's score even if it's 0.

#     Context: {context}

#     Examples:

#     Question: How far away is the moon?
#     Answer: The moon is 384,400 km away.
#     Score: 5

#     Question: How far away is the sun?
#     Answer: I don't know
#     Score: 0

#     Your turn!
#     Question: {question}"""
# )


# def get_answers(docs, question, llm):
#     answers_chain = answers_prompt | llm
#     return {
#         "question": question,
#         "answers": [
#             {
#                 "answer": answers_chain.invoke({"question": question, "context": doc.page_content}).content,
#                 "source": doc.metadata["source"],
#                 "data": doc.metadata["lastmod"],
#             }
#             for doc in docs
#         ],
#     }


# choose_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
# 		Use ONLY the following pre-existing answers to answer the user's question.
#         Use the answers that have the highest score (more helpful) and favor the most recent ones.
#         Cite sources and return the sources of the answers as they are, do not change them.
#         Answers: {answers}
# 		""",
#         ),
#         ("human", "{question}",),
#     ]
# )


# def choose_answer(answers, question, llm):
#     choose_chain = choose_prompt | llm
#     condensed = "\n\n".join(
#         f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['data']}\n"
#         for answer in answers
#     )
#     return choose_chain.invoke(
#         {
#             "question": question,
#             "answers": condensed,
#         }
#     )


# @st.cache_data
# def choose_answer(inputs):
#     answers = inputs["answers"]
#     question = inputs["question"]
#     llm = inputs["llm"]
#     choose_chain = choose_prompt | llm
#     condensed = "\n\n".join(
#         f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['data']}\n"
#         for answer in answers
#     )
#     return choose_chain.invoke(
#         {
#             "question": question,
#             "answers": condensed,
#         }
#     )


# def parse_page(soup):
#     return (
#         str(soup.find("main").get_text())
#         .replace("\n", " ")
#         .replace("\xa0", " ")
#         .replace("Edit page   Cloudflare DashboardDiscordCommunityLearning CenterSupport Portal  Cookie Settings", "")
#     )


# @retry(
#     wait=wait_random_exponential(min=1, max=60),
#     stop=stop_after_attempt(5),
#     retry=retry_if_exception_type(RateLimitError)
# )
# def embed_texts(texts, embeddings):
#     try:
#         embedded = embeddings.embed_documents(texts)
#         if not embedded or not all(isinstance(emb, list) for emb in embedded):
#             raise ValueError(
#                 f"Invalid embedding format returned from OpenAI API: {embedded[:2]}")
#         return embedded
#     except RateLimitError as e:
#         st.warning(f"Rate limit reached. Retrying in a moment... ({str(e)})")
#         raise


# def process_in_batches(texts, batch_size=10):
#     for i in range(0, len(texts), batch_size):
#         yield texts[i:i + batch_size]


# @st.cache_resource(show_spinner="웹사이트 로딩 중...")
# def load_website(url, key):
#     file_folder = './.cache/embeddings/site'

#     if not os.path.exists(file_folder):
#         os.makedirs(file_folder)
#     cache_dir = LocalFileStore(f"{file_folder}")
#     splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=1000,
#         chunk_overlap=200,
#     )
#     loader = SitemapLoader(
#         url,
#         parsing_function=parse_page,
#         filter_urls=[
#             'https://developers.cloudflare.com/ai-gateway/',
#             'https://developers.cloudflare.com/vectorize/',
#             'https://developers.cloudflare.com/workers-ai/',
#         ]
#     )
#     loader.requests_per_second = 0.5
#     docs = loader.load_and_split(text_splitter=splitter)
#     embeddings = OpenAIEmbeddings(api_key=key)
#     cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
#         embeddings, cache_dir)
#     vector_store = FAISS.from_documents(docs, cached_embeddings)
#     return vector_store.as_retriever()


# st.title("🌐 Site GPT")
# st.markdown("Welcome to Site GPT!")

# with st.sidebar:
#     st.markdown('<a href="https://github.com/GoGetShitDone/DOC_GPT" target="_blank"><button style="background-color:#0F1116;color:white;padding:10px 30px;border:none;border-radius:5px;cursor:pointer;">🍯 Ullala GitHub</button></a>', unsafe_allow_html=True)
#     api_key = st.text_input(
#         "OpenAI API Key", type="password", value=st.session_state.api_key)
#     go_button = st.button("Go")

# if api_key:
#     st.session_state.api_key = api_key
#     if is_valid_api_key(api_key):
#         st.session_state.llm = get_llm(api_key)
#         st.sidebar.success("API 키가 유효합니다.")
#     else:
#         st.sidebar.error("잘못된 API 키입니다. OpenAI API 키를 확인하고 다시 시도해주세요.")
#         st.session_state.llm = None
#         st.session_state.retriever = None

# if go_button and st.session_state.llm:
#     url = "https://developers.cloudflare.com/sitemap-0.xml"
#     try:
#         st.session_state.retriever = load_website(
#             url, st.session_state.api_key)
#         st.sidebar.success("웹사이트가 성공적으로 로드되었습니다.")
#     except Exception as e:
#         st.sidebar.error(f"웹사이트 로드 중 오류 발생: {str(e)}")
#         st.session_state.retriever = None

# if st.session_state.llm and st.session_state.retriever:
#     query = st.text_input("웹사이트에 대해 질문하세요:")
#     if query:
#         try:
#             docs = st.session_state.retriever.get_relevant_documents(query)
#             answers = get_answers(docs, query, st.session_state.llm)
#             result = choose_answer(
#                 answers["answers"], query, st.session_state.llm)
#             st.markdown(result.content)
#         except Exception as e:
#             st.error(f"질문 처리 중 오류 발생: {str(e)}")
#             st.error(traceback.format_exc())
# else:
#     st.info("OpenAI API 키를 입력한 후 'Go' 버튼을 클릭해주세요.")
