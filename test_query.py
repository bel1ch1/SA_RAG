import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings


# 1. Создаем обертку для SentenceTransformer
class FRIDAEmbeddingFunction:
    def __init__(self, model_name: str = "ai-forever/FRIDA"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

# 2. Инициализация с правильной embedding-функцией
collection_name = "documents"
chroma_path = "C:/work/SA_RAG/chroma_db2"

embedding = HuggingFaceEmbeddings(
    model_name="ai-forever/FRIDA"
)
client = chromadb.PersistentClient(path=chroma_path)

# 3. Создание vectorstore с правильной embedding-функцией
vectorstore = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=embedding,
    persist_directory=chroma_path
)

# 4. Создание retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 5. Пример запроса
query = "Какие бывают компоненты промежуточного контура?"
results = retriever.get_relevant_documents(query)

# Вывод результатов
for i, doc in enumerate(results, 1):
    print(f"Результат #{i}:")
    print(doc.page_content[:])  # Первые 200 символов
    print("-" * 80)
