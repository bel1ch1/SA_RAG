import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Инициализация компонентов
@st.cache_resource
def init_components():
    # 1. Инициализация LLM
    llm = ChatOpenAI(
        model=st.secrets.openrouter.model,
        openai_api_key=st.secrets.openrouter.api_key,
        openai_api_base=st.secrets.openrouter.api_base,
        temperature=0
    )

    # 2. Инициализация эмбеддера
    embedding = HuggingFaceEmbeddings(model_name="ai-forever/FRIDA")

    # 3. Подключение ChromaDB
    vectordb = Chroma(
        persist_directory=st.secrets.chroma.chroma_path,
        collection_name=st.secrets.chroma.collection_name,
        embedding_function=embedding
    )

    # 4. Настройка ретривера
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3,},
        search_type="similarity"
    )

    # 5. Кастомный промпт
    template = """
    You are an assistant for working with Sinamics S120. Answer the questions only in Russian, using the provided context.
    If you don't have the necessary information in the context, say you don't know the answer.

    Context: {context}

    Question: {question}

    Detailed and complete answer:"""

    qa_prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # 6. Создание RAG цепи
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )

    return llm, qa_chain

# Инициализация
llm, qa_chain = init_components()

# Инициализация состояния чата
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.use_rag = True
    st.session_state.show_sources = True

# Сайдбар с настройками
with st.sidebar:
    st.header("Настройки поиска")
    st.session_state.use_rag = st.checkbox(
        "Использовать базу знаний",
        value=st.session_state.use_rag
    )
    if st.session_state.use_rag:
        st.session_state.show_sources = st.checkbox(
            "Показывать источники",
            value=st.session_state.show_sources
        )

def format_sources(docs):
    sources = []
    for i, doc in enumerate(docs):
        source_name = doc.metadata.get("source", "")
        sources.append(
            f"##### 📄 Источник {i+1}: {source_name}\n"
            f"{doc.page_content}\n"
            f"---"
        )
    return "\n\n".join(sources)

# Отображение истории чата
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

        # Отображаем источники если они есть и включена настройка
        if (role == "assistant" and
            "source_docs" in message.additional_kwargs and
            st.session_state.show_sources):
            with st.expander("🔍 Показать полные источники"):
                st.markdown(format_sources(message.additional_kwargs["source_docs"]))

# Обработка запроса
if prompt := st.chat_input("Ваш вопрос о Sinamics S120..."):
    # Добавляем сообщение пользователя
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)

    # Отображаем сообщение пользователя
    with st.chat_message("user"):
        st.write(prompt)

    # Генерируем ответ
    with st.chat_message("assistant"):
        with st.spinner("Анализирую документацию..."):
            try:
                if st.session_state.use_rag:
                    # Полноценный RAG запрос
                    result = qa_chain({"query": prompt})

                    # Создаем объект ответа с полными источниками
                    response = AIMessage(
                        content=result["result"],
                        additional_kwargs={
                            "source_docs": result["source_documents"]
                        }
                    )
                else:
                    # Обычный режим
                    response = llm.invoke(st.session_state.messages)

                # Отображаем ответ
                st.write(response.content)

                # Показываем источники если включено
                if (st.session_state.use_rag and
                    st.session_state.show_sources and
                    "source_docs" in response.additional_kwargs):

                    with st.expander("🔍 Показать полные источники"):
                        st.markdown(format_sources(response.additional_kwargs["source_docs"]))

                # Добавляем в историю
                st.session_state.messages.append(response)

            except Exception as e:
                st.error("Ошибка обработки запроса")
                error_message = AIMessage(content="⚠️ Произошла ошибка, попробуйте позже")
                st.session_state.messages.append(error_message)
