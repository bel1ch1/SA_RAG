import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

OPENROUTER_API_KEY = st.secrets["openai"]["api_key"]
OPENROUTER_API_BASE = st.secrets["openai"]["api_base"]
OPENROUTER_MODEL = st.secrets["openai"]["model"]


# Настройки страницы
st.set_page_config(
    page_title="Чат-бот с OpenAI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# Инициализация модели
@st.cache_resource
def load_chat_model():
    return ChatOpenAI(
            model=OPENROUTER_MODEL,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_API_BASE,
        )

llm = load_chat_model()

# Инициализация истории чата
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение истории чата
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Обработка ввода пользователя
if prompt := st.chat_input("Напишите ваше сообщение..."):
    # Добавляем сообщение пользователя в историю
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)

    # Отображаем сообщение пользователя
    with st.chat_message("user"):
        st.write(prompt)

    # Генерируем ответ
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            response = llm.invoke(st.session_state.messages)
            st.write(response.content)

    # Добавляем ответ в историю
    st.session_state.messages.append(response)
