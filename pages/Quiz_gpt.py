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
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, Document

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
)


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="Quiz GPT",
    page_icon="🕹️",
    layout="wide",
)

st.title("🕹️ Quiz GPT")


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
def get_openai_model(api_key):
    return ChatOpenAI(
        temperature=0.1,
        model="gpt-4-0613",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_key=api_key,
    )


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


@st.cache_data(show_spinner="위키피디아 검색 중...")
def wiki_search(term):
    try:
        # 먼저 검색어와 정확히 일치하는 페이지를 찾습니다
        try:
            page = wikipedia.page(term)
            return [{"page_content": page.content, "metadata": {"title": page.title}}]
        except wikipedia.exceptions.DisambiguationError as e:
            # 검색어가 여러 가지를 의미할 수 있는 경우, 첫 번째 결과를 사용합니다
            page = wikipedia.page(e.options[0])
            return [{"page_content": page.content, "metadata": {"title": page.title}}]
        except wikipedia.exceptions.PageError:
            # 정확한 일치가 없는 경우, 검색을 수행합니다
            search_results = wikipedia.search(term, results=1)
            if not search_results:
                return []
            page = wikipedia.page(search_results[0])
            return [{"page_content": page.content, "metadata": {"title": page.title}}]
    except Exception as e:
        st.error(f"위키피디아 검색 중 오류 발생: {str(e)}")
        return []


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


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    당신은 선생님 역할을 하는 도움이 되는 조수입니다.

    주어진 컨텍스트만을 바탕으로 사용자의 지식을 테스트하기 위한 (열) 개의 질문을 만드세요.

    각 질문은 4개의 답변을 가져야 하며, 그 중 3개는 틀린 답변이고 1개는 정답이어야 합니다.

    정답을 표시하기 위해 (o)를 사용하세요.

    문제의 난이도는 {difficulty}입니다. "쉬움" 난이도의 경우, 컨텍스트에서 쉽게 추론할 수 있는 질문을 만드세요. "보통" 난이도의 경우, 60%의 질문은 쉽게 추론할 수 있고, 40%는 약간의 사고가 필요한 질문을 만드세요. "어려움" 난이도의 경우, 40%의 질문은 쉬움에서 보통 수준이고, 60%는 깊은 이해와 추론이 필요한 질문을 만드세요.

    질문 예시:

    질문: 바다의 색은 무엇인가요?
    답변: 빨강|노랑|초록|파랑(o)

    질문: 조지아의 수도는 어디인가요?
    답변: 바쿠|트빌리시(o)|마닐라|베이루트

    질문: 아바타는 언제 개봉했나요?
    답변: 2007|2001|2009(o)|1998

    질문: 줄리어스 시저는 누구였나요?
    답변: 로마 황제(o)|화가|배우|모델

    이제 시작하세요!

    컨텍스트: {context}
""",
        )
    ]
)

formatting_prompt = ChatPromptTemplate.from_messages([(
    "system",
    """
    당신은 강력한 포맷팅 알고리즘입니다.

    시험 문제를 JSON 형식으로 포맷팅합니다.
    (o)가 있는 답변이 정답입니다.

    입력 예시:
    질문: 바다의 색은 무엇인가요?
    답변: 빨강|노랑|초록|파랑(o)

    질문: 조지아의 수도는 어디인가요?
    답변: 바쿠|트빌리시(o)|마닐라|베이루트

    질문: 아바타는 언제 개봉했나요?
    답변: 2007|2001|2009(o)|1998

    질문: 줄리어스 시저는 누구였나요?
    답변: 로마 황제(o)|화가|배우|모델


    출력 예시:

    ```json
    {{ "questions": [
            {{
                "question": "바다의 색은 무엇인가요?",
                "answers": [
                        {{
                            "answer": "빨강",
                            "correct": false
                        }},
                        {{
                            "answer": "노랑",
                            "correct": false
                        }},
                        {{
                            "answer": "초록",
                            "correct": false
                        }},
                        {{
                            "answer": "파랑",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "조지아의 수도는 어디인가요?",
                "answers": [
                        {{
                            "answer": "바쿠",
                            "correct": false
                        }},
                        {{
                            "answer": "트빌리시",
                            "correct": true
                        }},
                        {{
                            "answer": "마닐라",
                            "correct": false
                        }},
                        {{
                            "answer": "베이루트",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "아바타는 언제 개봉했나요?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "줄리어스 시저는 누구였나요?",
                "answers": [
                        {{
                            "answer": "로마 황제",
                            "correct": true
                        }},
                        {{
                            "answer": "화가",
                            "correct": false
                        }},
                        {{
                            "answer": "배우",
                            "correct": false
                        }},
                        {{
                            "answer": "모델",
                            "correct": false
                        }},
                ]
            }}
        ]
    }}
    ```
    이제 시작하세요!
    문제들: {context}
""")])

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    docs = None
    topic = None
    choice = st.selectbox(
        "사용할 소스를 선택하세요.",
        ("위키피디아", "파일",),
    )

    if api_key:
        if is_valid_api_key(api_key):
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API 키가 유효합니다.")
            llm = get_openai_model(api_key)

            if choice == "위키피디아":
                topic = st.text_input("위키피디아 검색...")
                if topic:
                    docs = wiki_search(topic)
                    if docs:
                        st.success(
                            f"'{docs[0]['metadata']['title']}' 문서를 찾았습니다.")
                    else:
                        st.warning(
                            f"'{topic}'에 대한 검색 결과가 없습니다. 다른 검색어를 시도해보세요.")
            else:
                file = st.file_uploader("파일 업로드 (.docx, .txt, .pdf, .md)", type=[
                                        "pdf", "docx", "txt", "md",])
                if file:
                    docs = split_file(file)
                    st.write(f"파일에서 {len(docs)}개의 청크를 추출했습니다.")

            difficulty = st.selectbox("퀴즈 난이도 선택", ["쉬움", "보통", "어려움"])

            if st.button("퀴즈 생성"):
                if docs:
                    st.session_state.quiz = run_quiz_chain(
                        docs, topic if topic else file.name, _llm=llm, difficulty=difficulty)
                    st.session_state.answers = [
                        None] * len(st.session_state.quiz["questions"])
                    st.session_state.submitted = False
                st.success("퀴즈가 생성되었습니다!")
            else:
                st.error("주제를 입력하거나 파일을 업로드해주세요.")
        else:
            st.error("잘못된 API 키입니다. OpenAI API 키를 확인하고 다시 시도해주세요.")
    else:
        st.warning("OpenAI API 키를 입력해주세요.")

# # 디버그 정보 출력
# st.write("디버그 정보:")
# st.write(f"docs: {docs}")
# st.write(f"topic: {topic}")
# st.write(f"choice: {choice}")

# if 'quiz' in st.session_state:
#     st.write(f"quiz questions: {len(st.session_state.quiz['questions'])}")

if not docs:
    st.markdown(
        "Quiz GPT 를 활용해 문서 또는 Wiki로 문제를 만들 수 있습니다.<br>사이드바에 API 키를 입력하고 소스를 선택해주세요.",
        unsafe_allow_html=True)
elif api_key and is_valid_api_key(api_key):
    if 'quiz' in st.session_state and 'questions' in st.session_state.quiz and len(st.session_state.quiz["questions"]) > 0:
        st.subheader(f"주제: {topic if topic else '파일 업로드'}")
        if 'answers' not in st.session_state or len(st.session_state.answers) != len(st.session_state.quiz["questions"]):
            st.session_state.answers = [None] * \
                len(st.session_state.quiz["questions"])
        if 'submitted' not in st.session_state:
            st.session_state.submitted = False

        with st.form("questions_form"):
            correct_count = 0
            for i, question in enumerate(st.session_state.quiz["questions"]):
                st.write(question["question"])
                st.session_state.answers[i] = st.radio(
                    f"질문 {i+1}에 대한 답변을 선택하세요.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=f"question_{i}"
                )

                # 각 질문 바로 아래에 결과 메시지 표시
                if st.session_state.submitted:
                    if st.session_state.answers[i] is None:
                        st.warning(f"질문 {i+1}: 답변하지 않았습니다.")
                    elif {"answer": st.session_state.answers[i], "correct": True} in question["answers"]:
                        st.success(f"질문 {i+1}: 정답입니다!")
                        correct_count += 1
                    else:
                        st.error(f"질문 {i+1}: 틀렸습니다.")

                st.write("")  # 각 질문 사이에 약간의 공간 추가

            submitted = st.form_submit_button("제출")
            if submitted:
                st.session_state.submitted = True
                st.experimental_rerun()  # 폼 제출 후 페이지를 다시 로드하여 결과를 즉시 표시

        # 전체 결과 요약
        if st.session_state.submitted:
            st.write(
                f"{len(st.session_state.quiz['questions'])}개 중 {correct_count}개를 맞추셨습니다!")

            if correct_count == len(st.session_state.quiz['questions']):
                st.balloons()
                st.success("축하합니다! 모든 문제를 맞추셨습니다!")

    else:
        st.info("사이드바에서 '퀴즈 생성' 버튼을 클릭하여 새로운 퀴즈를 만드세요.")
