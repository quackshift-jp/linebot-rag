import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader

load_dotenv(verbose=True)


os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
FAISS_DB_DIR = "vectorstore"

llm_model = ChatOpenAI()


def load_text_documents(path: str) -> list[Document]:
    loader = DirectoryLoader(path=path, loader_cls=TextLoader, glob="*.txt")
    return loader.load()


def split_documents_to_chunk(documents: list[Document]) -> list[Document]:
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    return text_splitter.split_documents(documents)


def save_vector_store(chunked_documents: list[Document], save_dir: str) -> None:
    faiss_db = FAISS.from_documents(
        documents=chunked_documents, embedding=OpenAIEmbeddings()
    )
    faiss_db.save_local(save_dir)


def get_retrieval_chain(faiss_db: FAISS, llm_model: ChatOpenAI) -> RetrievalQA:
    retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    return RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=retriever,
        return_source_documents=True,
        input_key="prompt",
    )


def main(path: str, input: str) -> dict[str, str]:
    prompt = f"""
    あなたは、プロの相談回答者です。
    回答が見つからない場合は、これまでの経験によって回答してください。

    - [入力]
    {input}

    - [出力]
    """
    documents = load_text_documents(path=path)
    chunks = split_documents_to_chunk(documents)
    save_vector_store(chunked_documents=chunks, save_dir=FAISS_DB_DIR)
    retrieval_chain = get_retrieval_chain(
        faiss_db=FAISS.load_local(FAISS_DB_DIR, embeddings=OpenAIEmbeddings()),
        llm_model=llm_model,
    )
    response = retrieval_chain({"prompt": prompt})
    return response
