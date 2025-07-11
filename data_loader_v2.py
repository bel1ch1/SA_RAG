from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
import chromadb
from pathlib import Path

def load_data_to_chromadb(file_path: str, collection_name: str = "documents"):
    """
    Uploads data from a text file to ChromaDB in chunks of 256 characters.

    Args:
        file_path: the path to the text file
        collection_name: the name of the collection in ChromaDB (default is "documents")
    """
    # Creating a temporary directory for file processing
    temp_dir = Path("temp_rag")
    temp_dir.mkdir(exist_ok=True)

    # Copy the file to a temporary directory
    import shutil
    shutil.copy(file_path, temp_dir)

    # Initialize the ChromaDB client
    db = chromadb.PersistentClient(path="./chroma_db2")

    # Creating a collection
    chroma_collection = db.get_or_create_collection(collection_name)

    # Creating vector storage
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Configuring the parser to split into chunks of 256 characters
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separator=" ",
        paragraph_separator="\n\n"
    )

    # Uploading documents
    documents = SimpleDirectoryReader(temp_dir).load_data()

    # Embedder
    embed_model = HuggingFaceEmbedding(model_name="ai-forever/FRIDA")

    # Split into chunks and index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[node_parser],
        embed_model=embed_model,
        show_progress=True
    )

    # Deleting the temporary directory
    shutil.rmtree(temp_dir)

    print("The data has been successfully uploaded")
    return index

if __name__ == "__main__":
    file_path = "extracted_data/sinamics_1.txt"
    load_data_to_chromadb(file_path)
