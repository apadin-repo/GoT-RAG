import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"
CHROMA_PATH = "chroma"
SYSTEM_PROMPT = """You are a witty and enthusiastic Game of Thrones expert.
Your job is to answer questions about the Game of Thrones books and share fun, surprising facts about the characters.
Respond with energy, personality, and a touch of humor—like a true GoT superfan.

If you do not know the answer, just say so. Never make anything up."""

llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion:\n{question}")
])

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    callbacks=[StdOutCallbackHandler()]
)

if __name__ == "__main__":
    while True:
        user_input = input("\nAsk me about anything about Game of Thrones: ")
        
        if user_input.lower() == "exit":
            break

        result = conversation_chain.invoke({"question": user_input})
        answer = result["answer"]
        print("\nAnswer:", answer)