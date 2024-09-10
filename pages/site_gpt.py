import streamlit as st
import requests
import os
import pickle
import hashlib
import time
import traceback
import numpy as np
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai.error import RateLimitError
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

st.set_page_config(
    page_title="Site GPT",
    page_icon="🌐",
    layout="wide",
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

CACHE_DIR = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), '.cache', 'site_files')
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(url):
    return os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + '.pkl')


def save_to_cache(url, data):
    with open(get_cache_path(url), 'wb') as f:
        pickle.dump(data, f)


def load_from_cache(url, max_age=604800):
    cache_path = get_cache_path(url)
    if os.path.exists(cache_path):
        mod_time = os.path.getmtime(cache_path)
        if time.time() - mod_time < max_age:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    return None


@st.cache_data(show_spinner=False)
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


@st.cache_resource
def get_llm(api_key):
    return ChatOpenAI(temperature=0.1, openai_api_key=api_key)


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
    Question: {question}"""
)


def get_answers(docs, question, llm):
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke({"question": question, "context": doc.page_content}).content,
                "source": doc.metadata["source"],
                "data": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
		Use ONLY the following pre-existing answers to answer the user's question.
        Use the answers that have the highest score (more helpful) and favor the most recent ones.
        Cite sources and return the sources of the answers as they are, do not change them.
        Answers: {answers}
		""",
        ),
        ("human", "{question}",),
    ]
)


def choose_answer(answers, question, llm):
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['data']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


@st.cache_data
def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    llm = inputs["llm"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['data']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    return (
        str(soup.find("main").get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("Edit page   Cloudflare DashboardDiscordCommunityLearning CenterSupport Portal  Cookie Settings", "")
    )


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimitError)
)
def embed_texts(texts, embeddings):
    try:
        embedded = embeddings.embed_documents(texts)
        if not embedded or not all(isinstance(emb, list) for emb in embedded):
            raise ValueError(
                f"Invalid embedding format returned from OpenAI API: {embedded[:2]}")
        return embedded
    except RateLimitError as e:
        st.warning(f"Rate limit reached. Retrying in a moment... ({str(e)})")
        raise


def process_in_batches(texts, batch_size=10):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


@st.cache_resource(show_spinner="웹사이트 로딩 중...")
def load_website(url, key):
    file_folder = './.cache/embeddings/site'

    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    cache_dir = LocalFileStore(f"{file_folder}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        filter_urls=[
            'https://developers.cloudflare.com/ai-gateway/',
            'https://developers.cloudflare.com/vectorize/',
            'https://developers.cloudflare.com/workers-ai/',
        ]
    )
    loader.requests_per_second = 0.5
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(api_key=key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever()


st.title("🌐 Site GPT")
st.markdown("Welcome to Site GPT!")

with st.sidebar:
    st.markdown('<a href="https://github.com/GoGetShitDone/DOC_GPT" target="_blank"><button style="background-color:#0F1116;color:white;padding:10px 30px;border:none;border-radius:5px;cursor:pointer;">🍯 Ullala GitHub</button></a>', unsafe_allow_html=True)
    api_key = st.text_input(
        "OpenAI API Key", type="password", value=st.session_state.api_key)
    go_button = st.button("Go")

if api_key:
    st.session_state.api_key = api_key
    if is_valid_api_key(api_key):
        st.session_state.llm = get_llm(api_key)
        st.sidebar.success("API 키가 유효합니다.")
    else:
        st.sidebar.error("잘못된 API 키입니다. OpenAI API 키를 확인하고 다시 시도해주세요.")
        st.session_state.llm = None
        st.session_state.retriever = None

if go_button and st.session_state.llm:
    url = "https://developers.cloudflare.com/sitemap-0.xml"
    try:
        st.session_state.retriever = load_website(
            url, st.session_state.api_key)
        st.sidebar.success("웹사이트가 성공적으로 로드되었습니다.")
    except Exception as e:
        st.sidebar.error(f"웹사이트 로드 중 오류 발생: {str(e)}")
        st.session_state.retriever = None

if st.session_state.llm and st.session_state.retriever:
    query = st.text_input("웹사이트에 대해 질문하세요:")
    if query:
        try:
            docs = st.session_state.retriever.get_relevant_documents(query)
            answers = get_answers(docs, query, st.session_state.llm)
            result = choose_answer(
                answers["answers"], query, st.session_state.llm)
            st.markdown(result.content)
        except Exception as e:
            st.error(f"질문 처리 중 오류 발생: {str(e)}")
            st.error(traceback.format_exc())
else:
    st.info("OpenAI API 키를 입력한 후 'Go' 버튼을 클릭해주세요.")
