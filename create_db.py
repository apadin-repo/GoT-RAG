import os
import shutil
import nltk
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
import glob

# document loader (UnstructuredMarkdownLoader) depends on NLTK's pretrained models to process and chunk text accurately.
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma"
DATA_PATH = "got_books"

def load_documents():
    paths = glob.glob(f"{DATA_PATH}/*.md")
    documents = []
    for path in paths:
        try:
            loader = UnstructuredMarkdownLoader(path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"error loading {path}: {e}")
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def save_db(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )

def main():
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print("Splitting into chunks...")
    chunks = split_text(documents)
    print(f"Generated {len(chunks)} text chunks.")

    print("Saving to Chroma DB...")
    save_db(chunks)
    print(f"Saved to vector store at {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
