import os
import requests
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

st.set_page_config(
    page_title="Document GPT",
    page_icon="📄",
    layout="wide",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_resource
def get_openai_model(api_key):
    return ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        openai_api_key=api_key,
    )


@st.cache_data(show_spinner="Embedding File...")
def embed_file(file, api_key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


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


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


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


st.title("📄 Document GPT")

st.markdown("환영합니다!<br>이 챗봇을 사용하여 파일을 업로드 하고 AI에게 질문하세요!<br>사이드 바에 API 키를 입력하고 파일을 업로드하세요!<br>API 키는 저장되지 않습니다.",
            unsafe_allow_html=True)

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    file = st.file_uploader("Upload a .txt, .pdf, .docs, .md files only", type=[
                            "pdf", "txt", "docx", "md"])

if api_key:
    if is_valid_api_key(api_key):
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API 키가 유효합니다.")

        if file:
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
        else:
            st.warning("Please upload a file in the sidebar.")
    else:
        st.error("Invalid API key. Please check your OpenAI API key and try again.")
elif not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
