from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import streamlit as st
import time
import os

# 여기서 자신의 OpenAI api key로 바꿔주세요
#os.environ["OPENAI_API_KEY"] ="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"

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("OpenAI API를 먼저 입력해주세요.")
    st.stop()

import os
os.environ["OPENAI_API_KEY"] = openai_api_key

# temperature는 0에 가까워질수록 형식적인 답변을 내뱉고, 1에 가까워질수록 창의적인 답변을 내뱉음
llm = ChatOpenAI(temperature=0.2)

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("자주 쓰는 링크 정리.pdf")
pages = loader.load_and_split()

data = []
for content in pages:
    data.append(content)


# 올린 파일 내용 쪼개기
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

# 쪼갠 내용 vectorstore 데이터베이스에 업로드하기
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# 데이터베이스에 업로드 한 내용을 불러올 수 있도록 셋업
retriever = vectorstore.as_retriever()

# 에이전트가 사용할 내용 불러오는 툴 만들기
from langchain.agents.agent_toolkits import create_retriever_tool

tool = create_retriever_tool(
    retriever,
    "Role_in_evaluating_and_revising_resumes",
    "Evaluate and revise the contents of the customer resumes and then return the contents",
)
tools = [tool]

# 대화 내용 기록하는 메모리 변수 셋업
memory_key = "history"

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)

memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder

# AI 에이전트가 사용할 프롬프트 짜주기
system_message = SystemMessage(
    content=(
        "You evaluate and revise the contents of the customer's resumes and then return the contents."
        "Do your best to answer the questions."
        "Feel free to use any tools available to look up "
        "relevant information, only if necessary"
        "Do not generate false answers to questions that are not related to 자소서바이블.pdf."
        "If you don't know the answer, just say that you don't know. Don't try to make up an answer."
        """내 자소서를 비레체로 평가해주세요라고 하면 자소서바이블.pdf 2페이지 내용을 기반으로 아래의 예시처럼 체크해준다.
        1. ‘질문에 대한 답’을 하고 있는가?
          // 평가 결과: 만족 또는 미흡
          // 평가 이유: ㅇㅇㅇㅇ ////
        2. 결론이 ‘근거’를 가지고 있는가?
          // 평가 결과: 만족 또는 미흡
          // 평가 이유: ㅇㅇㅇㅇ ////
        3. 소제목 또는 첫 문장이 ‘요약과 압축’이 되어있는가?
          // 평가 결과: 만족 또는 미흡
          // 평가 이유: ㅇㅇㅇㅇ ////
        4. 말하고자 하는 결론이 ‘서두에 배치’되어 있는가?
          // 평가 결과: 만족 또는 미흡
          // 평가 이유: ㅇㅇㅇㅇ ////
        5. 근거가 직무,산업,직장 중 한 가지와 ‘연결’되었는가?
          // 평가 결과: 만족 또는 미흡
          // 평가 이유: ㅇㅇㅇㅇ ////
        6. 한 개의 문단 또는 답변에서 ‘한 개의 메시지’로 답변했는가?
          // 평가 결과: 만족 또는 미흡
          // 평가 이유: ㅇㅇㅇㅇ ////
        7. 문장을 최대한 ‘짧게 구성’했는가?
          // 평가 결과: 만족 또는 미흡
          // 평가 이유: ㅇㅇㅇㅇ
        """
        "Make sure to answer in Korean."
    )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
)

# 에이전트 셋업해주기
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

from langchain.agents import AgentExecutor

# 위에서 만든 툴, 프롬프트를 토대로 에이전트 실행시켜주기 위해 셋업
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)

# 웹사이트 제목
st.title("형PT")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 웹사이트에서 유저의 인풋을 받고 위에서 만든 AI 에이전트 실행시켜서 답변 받기
if prompt := st.chat_input("내 자소서를 비레체로 평가해주세요."):

# 유저가 보낸 질문이면 유저 아이콘과 질문 보여주기
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# AI가 보낸 답변이면 AI 아이콘이랑 LLM 실행시켜서 답변 받고 스트리밍해서 보여주기
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        result = agent_executor({"input": prompt})
        for chunk in result["output"].split():
            full_response += chunk + " "
            time.sleep(0.5)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
