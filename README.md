# GoT-RAG 
**A Game of Thrones Retrieval-Augmented Generation (RAG) chatbot powered by LangChain + OpenAI**

This project allows you to ask questions about the *Game of Thrones* books. It uses a RAG pipeline to retrieve context from markdown files and generate intelligent, fun, and witty answers with OpenAI's GPT models.

---

## ⚙️ Requirements

- Python 3.10+
- `langchain`
- `langchain-openai`
- `langchain-chroma`
- `langchain-community`
- `python-dotenv`
- `openai`
- `chromadb`
- *(Optional)*: `nltk` if you're using advanced Markdown chunking

Install dependencies:
```bash
pip install -r requirements.txt
```

Create `.env`:
```
OPENAI_API_KEY=your_openai_key_here
```

---

## Setup Steps

### 1. Add Your Markdown Files
Unzip and palace GoT-related `.md` documents inside the `got_books/` folder. `The Feast at Hollow Ridge` is about my dogs Frankie and Tego. 

### 2. Create the Vector DB
Run this to load, chunk, and embed documents:
```bash
python create_db.py
```

This creates a local Chroma DB in the `chroma/` directory.

### 3. Start the Chatbot
```bash
python app.py
```

You'll be prompted to ask questions. Type `exit` to quit the app.

---

## How It Works

- **Document Loading**: Markdown files are loaded using `UnstructuredMarkdownLoader`
- **Text Splitting**: Documents are split into overlapping chunks with `RecursiveCharacterTextSplitter`
- **Embeddings**: Chunks are embedded via `OpenAIEmbeddings`
- **Vector DB**: Embeddings are stored in `Chroma` (persisted locally)
- **Chat Engine**: `ConversationalRetrievalChain` combines retrieval with LLM response
- **Memory**: Maintains conversation context with `ConversationBufferMemory`
- **Prompting**: A custom system prompt gives the bot a witty GoT expert personality

---

## 📌 Example Prompt

```
Ask me about anything about Game of Thrones: Who is Jon Snow's real mother?

Answer: Ah, the mysterious case of Jon Snow's mother! This topic has sparked endless debates and theories among Game of Thrones fans. In the books, Jon's mother is not explicitly revealed, but there are many hints and clues scattered throughout the story. Ned Stark's reluctance to talk about Jon's mother and his refusal to send Jon away indicate that there is more to Jon's parentage than meets the eye. Some fans believe that Jon's mother could be Ned's sister, Lyanna Stark, and that his father might be Rhaegar Targaryen, making Jon a legitimate heir to the Iron Throne. However, as of now, Jon Snow's true parentage remains one of the biggest mysteries in the Game of Thrones universe. So, buckle up, my fellow Thrones enthusiasts, the truth may just be around the corner in the upcoming books! Valar Morghulis!
```