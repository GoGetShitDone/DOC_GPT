import streamlit as st
import time
import openai
import json
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper


class OpenAIAssistant:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        self.assistant = None
        self.thread = None

    def create_assistant(self):
        try:
            self.assistant = openai.beta.assistants.create(
                name='Investor Assistant',
                instructions='You are an AI assistant specializing in investment analysis. Analyze companies, their financial performance, market position, and future prospects. Use DuckDuckGo and Wikipedia to gather information when needed.',
                model='gpt-4-0125-preview',
                tools=self.get_functions()
            )
            return self.assistant
        except Exception as e:
            st.error(f"Assistant 생성 중 오류 발생: {str(e)}")
            return None

    def get_functions(self):
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'DuckDuckGoSearchTool',
                    'description': 'Use this tool to search for current information using DuckDuckGo.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The search query.',
                            }
                        },
                        'required': ['query'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'WikipediaSearchTool',
                    'description': 'Use this tool to search for general information using Wikipedia.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The search query.',
                            },
                        },
                        'required': ['query'],
                    },
                },
            },
        ]

    def create_thread(self):
        if not self.thread:
            self.thread = openai.beta.threads.create()
        return self.thread

    def send_message(self, content, role="user"):
        return openai.beta.threads.messages.create(
            thread_id=self.thread.id, role=role, content=content
        )

    def create_run(self, question):
        return openai.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
        )

    def get_run(self, run_id):
        return openai.beta.threads.runs.retrieve(
            run_id=run_id,
            thread_id=self.thread.id,
        )

    def get_messages(self):
        messages = openai.beta.threads.messages.list(thread_id=self.thread.id)
        return list(messages)

    def submit_tool_outputs(self, run_id):
        run = self.get_run(run_id)
        if run.required_action:
            outputs = []
            for action in run.required_action.submit_tool_outputs.tool_calls:
                function_name = action.function.name
                arguments = json.loads(action.function.arguments)
                output = self.functions_map[function_name](arguments)
                outputs.append({"tool_call_id": action.id, "output": output})

            openai.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread.id,
                run_id=run_id,
                tool_outputs=outputs
            )

    @staticmethod
    def DuckDuckGoSearchTool(inputs):
        query = inputs['query']
        ddg = DuckDuckGoSearchRun()
        return ddg.run(query)

    @staticmethod
    def WikipediaSearchTool(inputs):
        query = inputs['query']
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wikipedia.invoke(query)

    functions_map = {
        'DuckDuckGoSearchTool': DuckDuckGoSearchTool,
        'WikipediaSearchTool': WikipediaSearchTool,
    }


class InvestorGPTApp:
    def __init__(self):
        self.setup_page()
        self.assistant = None

    @staticmethod
    def setup_page():
        st.set_page_config(
            page_title="Resrarch & Invest",
            page_icon="📈",
            layout="wide",
        )

    def run(self):
        st.title("📈 Resrarch & Invest")
        self.sidebar()
        self.main_content()

    def sidebar(self):
        with st.sidebar:
            st.markdown('<a href="https://github.com/GoGetShitDone/DOC_GPT" target="_blank"><button style="background-color:#0F1116;color:white;padding:10px 24px;border:none;border-radius:5px;cursor:pointer;">🍯 Ullala GitHub</button></a>', unsafe_allow_html=True)
            api_key = st.text_input("OpenAI API Key", type="password")
            if api_key:
                if self.is_api_key_valid(api_key):
                    st.success("API 키가 유효합니다.")
                    self.assistant = OpenAIAssistant(api_key)
                    self.assistant.create_assistant()
                    self.assistant.create_thread()
                else:
                    st.error("잘못된 API 키입니다. 다시 확인해주세요.")

    def main_content(self):
        st.markdown(
            "##### 투자를 위한 리서치 GPT 입니다. API 키를 입력하고 궁금한 사항을 알려주세요.")

        if self.assistant:
            if 'messages' not in st.session_state:
                st.session_state['messages'] = []

            for message in st.session_state['messages']:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            query = st.chat_input("리서치가 필요한 회사에 대해 질문해주세요...")

            if query:
                self.process_query(query)
        else:
            st.warning("OpenAI API 키를 입력해주세요.")

    def process_query(self, query):
        st.session_state['messages'].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("리서치 중..."):
                try:
                    self.assistant.send_message(query)
                    run = self.assistant.create_run(query)

                    while run.status not in ["completed", "failed"]:
                        run = self.assistant.get_run(run.id)
                        if run.status == "requires_action":
                            self.assistant.submit_tool_outputs(run.id)
                        time.sleep(1)

                    if run.status == "completed":
                        messages = self.assistant.get_messages()
                        assistant_message = next(
                            msg for msg in messages if msg.role == "assistant")
                        response = assistant_message.content[0].text.value
                        st.markdown(response)
                        st.session_state['messages'].append(
                            {"role": "assistant", "content": response})
                    elif run.status == "failed":
                        st.error("리서치 생성에 실패했습니다. 다시 시도해주세요.")
                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")

    @staticmethod
    def is_api_key_valid(api_key):
        try:
            openai.api_key = api_key
            openai.models.list()
            return True
        except:
            return False


if __name__ == "__main__":
    app = InvestorGPTApp()
    app.run()

# import streamlit as st
# import time
# import openai
# import json
# from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
# from langchain.utilities import WikipediaAPIWrapper

# # 페이지 설정
# st.set_page_config(
#     page_title="Investor Research GPT",
#     page_icon="📈",
#     layout="wide",
# )


# # API 키 유효성 검사 함수
# def is_api_key_valid(api_key):
#     try:
#         openai.api_key = api_key
#         openai.models.list()
#         return True
#     except:
#         return False


# # Assistant 생성 함수
# def make_assistant(api_key):
#     try:
#         openai.api_key = api_key
#         assistant = openai.beta.assistants.create(
#             name='Investor Assistant',
#             instructions='You are an AI assistant specializing in investment analysis. Analyze companies, their financial performance, market position, and future prospects. Use DuckDuckGo and Wikipedia to gather information when needed.',
#             model='gpt-4o-mini',
#             tools=functions,
#         )
#         return assistant
#     except Exception as e:
#         st.error(f"Assistant 생성 중 오류 발생: {str(e)}")
#         return None


# # 검색 도구 함수
# def DuckDuckGoSearchTool(inputs):
#     query = inputs['query']
#     ddg = DuckDuckGoSearchRun()
#     return ddg.run(query)


# def WikipediaSearchTool(inputs):
#     query = inputs['query']
#     wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
#     return wikipedia.invoke(query)


# functions_map = {
#     'DuckDuckGoSearchTool': DuckDuckGoSearchTool,
#     'WikipediaSearchTool': WikipediaSearchTool,
# }


# functions = [
#     {
#         'type': 'function',
#         'function': {
#             'name': 'DuckDuckGoSearchTool',
#             'description': 'Use this tool to search for current information using DuckDuckGo.',
#             'parameters': {
#                 'type': 'object',
#                 'properties': {
#                     'query': {
#                         'type': 'string',
#                         'description': 'The search query.',
#                     }
#                 },
#                 'required': ['query'],
#             },
#         },
#     },
#     {
#         'type': 'function',
#         'function': {
#             'name': 'WikipediaSearchTool',
#             'description': 'Use this tool to search for general information using Wikipedia.',
#             'parameters': {
#                 'type': 'object',
#                 'properties': {
#                     'query': {
#                         'type': 'string',
#                         'description': 'The search query.',
#                     },
#                 },
#                 'required': ['query'],
#             },
#         },
#     },
# ]


# # Thread 및 Run 관련 함수
# def make_thread():
#     if 'thread' not in st.session_state or not st.session_state['thread']:
#         thread = openai.beta.threads.create()
#         st.session_state['thread'] = thread
#     return st.session_state['thread']


# def make_run(assistant_id, thread_id, question):
#     run = openai.beta.threads.runs.create(
#         thread_id=thread_id,
#         assistant_id=assistant_id,
#     )
#     return run


# def get_run(run_id, thread_id):
#     return openai.beta.threads.runs.retrieve(
#         run_id=run_id,
#         thread_id=thread_id,
#     )


# def send_message(thread_id, content, role="user"):
#     return openai.beta.threads.messages.create(
#         thread_id=thread_id, role=role, content=content
#     )


# def get_messages(thread_id):
#     messages = openai.beta.threads.messages.list(thread_id=thread_id)
#     return list(messages)


# def submit_tool_outputs(run_id, thread_id):
#     run = get_run(run_id, thread_id)
#     if run.required_action:
#         outputs = []
#         for action in run.required_action.submit_tool_outputs.tool_calls:
#             function_name = action.function.name
#             arguments = json.loads(action.function.arguments)
#             output = functions_map[function_name](arguments)
#             outputs.append({"tool_call_id": action.id, "output": output})

#         openai.beta.threads.runs.submit_tool_outputs(
#             thread_id=thread_id,
#             run_id=run_id,
#             tool_outputs=outputs
#         )


# # UI 구성
# st.title("📈 Investor GPT")

# with st.sidebar:
#     st.markdown('<a href="https://github.com/GoGetShitDone/DOC_GPT" target="_blank"><button style="background-color:#0F1116;color:white;padding:10px 24px;border:none;border-radius:5px;cursor:pointer;">🍯 Ullala GitHub</button></a>', unsafe_allow_html=True)
#     api_key = st.text_input("OpenAI API Key", type="password")
#     api_key_valid = False
#     if api_key:
#         if is_api_key_valid(api_key):
#             st.success("API 키가 유효합니다.")
#             api_key_valid = True
#         else:
#             st.error("잘못된 API 키입니다. 다시 확인해주세요.")

# st.markdown("##### 투자를 위한 리서치 GPT 입니다. API 키를 입력하고 회사나 얻고싶은 정보에 대해 알려주세요.")

# if api_key_valid:
#     if 'assistant' not in st.session_state or st.session_state['assistant'] is None:
#         with st.spinner("Assistant를 초기화 중..."):
#             st.session_state['assistant'] = make_assistant(api_key)

#     if st.session_state['assistant'] is None:
#         st.error("Assistant 초기화에 실패했습니다. API 키를 확인하고 다시 시도해주세요.")
#     else:
#         thread = make_thread()

#         if 'messages' not in st.session_state:
#             st.session_state['messages'] = []

#         for message in st.session_state['messages']:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#         query = st.chat_input("리서치가 필요한 회사에 대해 질문해주세요...")

#         if query:
#             st.session_state['messages'].append(
#                 {"role": "user", "content": query})
#             with st.chat_message("user"):
#                 st.markdown(query)

#             with st.chat_message("assistant"):
#                 message_placeholder = st.empty()
#                 message_placeholder.markdown("리서치 중...")

#                 try:
#                     send_message(thread.id, query)
#                     run = make_run(
#                         st.session_state['assistant'].id, thread.id, query)

#                     while run.status not in ["completed", "failed"]:
#                         run = get_run(run.id, thread.id)
#                         if run.status == "requires_action":
#                             submit_tool_outputs(run.id, thread.id)
#                         time.sleep(1)

#                     if run.status == "completed":
#                         messages = get_messages(thread.id)
#                         assistant_message = next(
#                             msg for msg in messages if msg.role == "assistant")
#                         response = assistant_message.content[0].text.value
#                         message_placeholder.markdown(response)
#                         st.session_state['messages'].append(
#                             {"role": "assistant", "content": response})
#                     elif run.status == "failed":
#                         message_placeholder.error("리서치 생성에 실패했습니다. 다시 시도해주세요.")
#                 except Exception as e:
#                     message_placeholder.error(f"오류 발생: {str(e)}")

# else:
#     st.warning("OpenAI API 키를 입력해주세요.")

# # 이전 과제의 코드
# # import streamlit as st
# # import os
# # import requests
# # from typing import Type
# # from langchain.chat_models import ChatOpenAI
# # from langchain.tools import BaseTool
# # from pydantic import BaseModel, Field
# # from langchain.agents import initialize_agent, AgentType
# # from langchain.utilities import DuckDuckGoSearchAPIWrapper
# # from langchain.schema import SystemMessage


# # def is_valid_api_key(api_key, api_type):
# #     if api_type == "openai":
# #         url = "https://api.openai.com/v1/engines"
# #         headers = {"Authorization": f"Bearer {api_key}"}
# #         response = requests.get(url, headers=headers)
# #         return response.status_code == 200
# #     elif api_type == "alphavantage":
# #         url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey={api_key}"
# #         response = requests.get(url)
# #         data = response.json()
# #         return "Symbol" in data and "Error Message" not in data
# #     return False


# # def get_openai_model(api_key):
# #     """유효한 API 키로 OpenAI 모델을 초기화합니다."""
# #     return ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-1106", openai_api_key=api_key)


# # llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-1106")

# # alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")


# # class StockMarketSymbolSearchToolArgsSchema(BaseModel):
# #     query: str = Field(
# #         description="The query you will search for.Example query: Stock Market Symbol for Apple Company"
# #     )


# # class StockMarketSymbolSearchTool(BaseTool):
# #     name = "StockMarketSymbolSearchTool"
# #     description = """
# #     Use this tool to find the stock market symbol for a company.
# #     It takes a query as an argument.

# #     """
# #     args_schema: Type[
# #         StockMarketSymbolSearchToolArgsSchema
# #     ] = StockMarketSymbolSearchToolArgsSchema

# #     def _run(self, query):
# #         ddg = DuckDuckGoSearchAPIWrapper()
# #         return ddg.run(query)


# # class CompanyOverviewArgsSchema(BaseModel):
# #     symbol: str = Field(
# #         description="Stock symbol of the company.Example: AAPL,TSLA",
# #     )


# # class CompanyOverviewTool(BaseTool):
# #     name = "CompanyOverview"
# #     description = """
# #     Use this to get an overview of the financials of the company.
# #     You should enter a stock symbol.
# #     """
# #     args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

# #     def _run(self, symbol):
# #         r = requests.get(
# #             f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
# #         )
# #         return r.json()


# # class CompanyIncomeStatementTool(BaseTool):
# #     name = "CompanyIncomeStatement"
# #     description = """
# #     Use this to get the income statement of a company.
# #     You should enter a stock symbol.
# #     """
# #     args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

# #     def _run(self, symbol):
# #         r = requests.get(
# #             f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
# #         )
# #         return r.json()["annualReports"]


# # class CompanyStockPerformanceTool(BaseTool):
# #     name = "CompanyStockPerformance"
# #     description = """
# #     Use this to get the weekly performance of a company stock.
# #     You should enter a stock symbol.
# #     """
# #     args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

# #     def _run(self, symbol):
# #         r = requests.get(
# #             f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
# #         )
# #         response = r.json()
# #         return list(response["Weekly Time Series"].items())[:200]


# # agent = initialize_agent(
# #     llm=llm,
# #     verbose=True,
# #     agent=AgentType.OPENAI_FUNCTIONS,
# #     handle_parsing_errors=True,
# #     tools=[
# #         CompanyIncomeStatementTool(),
# #         CompanyStockPerformanceTool(),
# #         StockMarketSymbolSearchTool(),
# #         CompanyOverviewTool(),
# #     ],
# #     agent_kwargs={
# #         "system_message": SystemMessage(
# #             content="""
# #             You are a hedge fund manager.

# #             You evaluate a company and provide your opinion and reasons why the stock is a buy or not.

# #             Consider the performance of a stock, the company overview and the income statement.

# #             Be assertive in your judgement and recommend the stock or advise the user against it.
# #         """
# #         )
# #     },
# # )

# # st.set_page_config(
# #     page_title="Investor GPT",
# #     page_icon="📈",
# #     layout="wide",
# # )

# # with st.sidebar:
# #     st.markdown('<a href="https://github.com/GoGetShitDone/DOC_GPT" target="_blank"><button style="background-color:#0F1116;color:white;padding:10px 24px;border:none;border-radius:5px;cursor:pointer;">🍯 Ullala GitHub</button></a>', unsafe_allow_html=True)

# #     openai_api_key = st.text_input("OpenAI API Key", type="password")
# #     openai_key_valid = False
# #     if openai_api_key:
# #         if is_valid_api_key(openai_api_key, "openai"):
# #             st.success("OpenAI API 키가 유효합니다.")
# #             os.environ["OPENAI_API_KEY"] = openai_api_key
# #             openai_key_valid = True
# #         else:
# #             st.error("잘못된 OpenAI API 키입니다. 다시 확인해주세요.")

# #     alpha_vantage_api_key = st.text_input(
# #         "Alpha Vantage API Key", type="password")
# #     alpha_vantage_key_valid = False
# #     if alpha_vantage_api_key:
# #         if is_valid_api_key(alpha_vantage_api_key, "alphavantage"):
# #             st.success("Alpha Vantage API 키가 유효합니다.")
# #             os.environ["ALPHA_VANTAGE_API_KEY"] = alpha_vantage_api_key
# #             alpha_vantage_key_valid = True
# #         else:
# #             st.error("잘못된 Alpha Vantage API 키입니다. 다시 확인해주세요.")


# # st.title("📈 Investor GPT")
# # st.markdown(
# #     "##### <br>투자를 위한 검색 GPT 입니다.<br><br>사이드바에 API 키를 입력하고 검색 회사를 입력해주세요.",
# #     unsafe_allow_html=True)

# # # 메인 영역에 회사 이름 입력 필드 추가
# # company = st.text_input("", placeholder="관심있는 회사 이름 검색")
# # if company and openai_key_valid and alpha_vantage_key_valid:
# #     llm = get_openai_model(openai_api_key)
# #     agent = initialize_investor_agent(llm)

# #     with st.spinner('Analyzing the company...'):
# #         result = agent.invoke(company)
# #         st.write(result["output"].replace("$", "\$"))
# # elif company:
# #     st.warning(
# #         "Please ensure both API keys are valid before searching for a company.")
