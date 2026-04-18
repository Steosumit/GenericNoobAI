"""
This file contains the RAG retriever function components
1. we will load the docs
2. do chunking
3. create embeddings
4. store the embeddings in a vector storage

Keep reading the doc as we go along:
https://docs.langchain.com/oss/python/integrations/document_loaders/
"""

import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from dotenv import load_dotenv

load_dotenv()  # load global env vars

# main class: to organize all the functions
class CustomRetriever():
    """This class is the main class to arrange the functions"""

    def __init__(self, file_path: str):
        # check if the directory exists else creat it
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print("Created /data directory")
        else:
            print("/data already exists")
        self.file_path = r"D:\Work\Projects\SteoNoteGent\data\sample.csv"  # file path list
        self.storage = None  # storage for documents objects
        self. retriever = None  # retriever object
    # loader: create the loader function as the RAG retriever
    def load(self):
        loader = CSVLoader(self.file_path)
        documents = loader.load()  # lsit of Documents objects, each row is a Document object
        print("[MSG] Documents loaded(Rows):")
        print(documents[0])
        self.storage = documents

    # chunking: split the document into smaller chunks
    def split(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        documents_splits = text_splitter.split_documents(self.storage)
        print(f"[MSG] Number of chunks created: {len(documents_splits)}")
        print("[MSG] Sample chunk:")
        print("===")
        print(documents_splits[0])
        print("===")
        self.storage = documents_splits  # why using common storage? Saves memory plus the emdedding are memory based

    # Embed: create embeddings of the chunks (self.storage) and store the vectors
    def embed_storage(self):
        embeder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = InMemoryVectorStore(embedding=embeder)  # ony embedder is needed as it is in memory
        vectorstore.add_documents(self.storage)
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
    def run(self)-> VectorStoreRetriever:
        self.load()
        self.split()
        self.embed_storage()
        return self.retriever

# test code
if __name__ == "__main__":
    retriever_instance = CustomRetriever("sample_data.csv")
    retriever = retriever_instance.run()
    print("[MSG] Retriever is ready to use.")
    # test the retriever
    query = "What are the times ?"
    result = retriever.invoke(query)
    print("\n\nRetrieval Results:", result)  # the best matching chunks(not the llm result that code is in app.py)

