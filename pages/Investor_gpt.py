import streamlit as st
import os
import requests
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import SystemMessage


def is_valid_api_key(api_key, api_type):
    if api_type == "openai":
        url = "https://api.openai.com/v1/engines"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers)
        return response.status_code == 200
    elif api_type == "alphavantage":
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        return "Symbol" in data and "Error Message" not in data
    return False


def get_openai_model(api_key):
    """유효한 API 키로 OpenAI 모델을 초기화합니다."""
    return ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-1106", openai_api_key=api_key)


llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-1106")

alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query you will search for.Example query: Stock Market Symbol for Apple Company"
    )


class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
    Use this tool to find the stock market symbol for a company.
    It takes a query as an argument.
    
    """
    args_schema: Type[
        StockMarketSymbolSearchToolArgsSchema
    ] = StockMarketSymbolSearchToolArgsSchema

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)


class CompanyOverviewArgsSchema(BaseModel):
    symbol: str = Field(
        description="Stock symbol of the company.Example: AAPL,TSLA",
    )


class CompanyOverviewTool(BaseTool):
    name = "CompanyOverview"
    description = """
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()


class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description = """
    Use this to get the income statement of a company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()["annualReports"]


class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockPerformance"
    description = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        response = r.json()
        return list(response["Weekly Time Series"].items())[:200]


agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        CompanyIncomeStatementTool(),
        CompanyStockPerformanceTool(),
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            You are a hedge fund manager.
            
            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
            
            Consider the performance of a stock, the company overview and the income statement.
            
            Be assertive in your judgement and recommend the stock or advise the user against it.
        """
        )
    },
)

st.set_page_config(
    page_title="Investor GPT",
    page_icon="📈",
    layout="wide",
)

with st.sidebar:
    st.markdown('<a href="https://github.com/GoGetShitDone/DOC_GPT" target="_blank"><button style="background-color:#0F1116;color:white;padding:10px 24px;border:none;border-radius:5px;cursor:pointer;">🍯 Ullala GitHub</button></a>', unsafe_allow_html=True)

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    openai_key_valid = False
    if openai_api_key:
        if is_valid_api_key(openai_api_key, "openai"):
            st.success("OpenAI API 키가 유효합니다.")
            os.environ["OPENAI_API_KEY"] = openai_api_key
            openai_key_valid = True
        else:
            st.error("잘못된 OpenAI API 키입니다. 다시 확인해주세요.")

    alpha_vantage_api_key = st.text_input(
        "Alpha Vantage API Key", type="password")
    alpha_vantage_key_valid = False
    if alpha_vantage_api_key:
        if is_valid_api_key(alpha_vantage_api_key, "alphavantage"):
            st.success("Alpha Vantage API 키가 유효합니다.")
            os.environ["ALPHA_VANTAGE_API_KEY"] = alpha_vantage_api_key
            alpha_vantage_key_valid = True
        else:
            st.error("잘못된 Alpha Vantage API 키입니다. 다시 확인해주세요.")


st.title("📈 Investor GPT")
st.markdown(
    "##### <br>투자를 위한 검색 GPT 입니다.<br><br>사이드바에 API 키를 입력하고 검색 회사를 입력해주세요.",
    unsafe_allow_html=True)

# 메인 영역에 회사 이름 입력 필드 추가
company = st.text_input("", placeholder="관심있는 회사 이름 검색")
if company and openai_key_valid and alpha_vantage_key_valid:
    llm = get_openai_model(openai_api_key)
    agent = initialize_investor_agent(llm)

    with st.spinner('Analyzing the company...'):
        result = agent.invoke(company)
        st.write(result["output"].replace("$", "\$"))
elif company:
    st.warning(
        "Please ensure both API keys are valid before searching for a company.")
