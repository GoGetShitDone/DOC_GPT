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

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
)

st.set_page_config(
    page_title="Document GPT",
    page_icon="📄",
    layout="wide",
)


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

st.markdown("##### <br>챗봇을 사용하여 파일을 업로드 하고 AI에게 질문하세요!<br><br>사이드 바에 API 키를 입력하고 파일을 업로드하세요!<br><br>API 키는 저장되지 않습니다.",
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<a href="https://github.com/GoGetShitDone/DOC_GPT" target="_blank"><button style="background-color:#0F1116;color:white;padding:10px 30px;border:none;border-radius:5px;cursor:pointer;">🍯 Ullala GitHub</button></a>', unsafe_allow_html=True)
    api_key = st.text_input("OpenAI API Key", type="password")
    file = st.file_uploader("Upload a .txt, .pdf, .docs, .md files only", type=[
                            "pdf", "txt", "docx", "md"])

    if api_key:
        if is_valid_api_key(api_key):
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API 키가 유효합니다.")

            if file:
                try:
                    retriever = embed_file(file, api_key)
                    llm = get_openai_model(api_key)
                    send_message("Good! Ask me anything!", "ai", save=False)
                    paint_history()
                    message = st.chat_input("첨부 자료에 관한 질문을 해주세요.")
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
                st.warning("파일을 업로드 해주세요.")
        else:
            st.error(
                "잘못된 API 키입니다. OpenAI API 키를 확인하고 다시 시도해주세요.")
    elif not api_key:
        st.warning("OpenAI API 키를 입력해주세요.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
