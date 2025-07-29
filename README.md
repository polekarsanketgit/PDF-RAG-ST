
# 📚 RAG Chatbot – PDF Question Answering with LangChain + Streamlit

This Streamlit app allows users to upload a PDF and interact with its contents using a conversational AI chatbot powered by OpenAI's GPT-4 and LangChain. It performs **Retrieval-Augmented Generation (RAG)** by embedding your document and fetching relevant context to answer natural language questions.

---

## 🚀 Features

- 📄 Upload a PDF (Max 300KB)
- 🔍 Ask questions based on the document content
- 🧠 Maintains conversation context with memory
- ⚡ Uses FAISS for vector search and SentenceTransformers for embeddings
- 🔐 Secure OpenAI API Key input (via sidebar)

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [SentenceTransformers](https://www.sbert.net/)
- [OpenAI GPT-4](https://platform.openai.com/docs/models/gpt-4)

---

## 📦 Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/rag-pdf-chatbot.git
   cd rag-pdf-chatbot
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```txt
   streamlit
   langchain
   langchain-community
   langchain-openai
   faiss-cpu
   sentence-transformers
   pymupdf
   ```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

> Replace `app.py` with your Python file name if different.

---

## 🔐 How to Use

1. Launch the app with Streamlit.
2. In the **sidebar**, enter your **OpenAI API Key**.
3. Upload a **PDF file** (less than 300KB).
4. Ask questions in the text input and get answers based on the PDF content.

---

## 📌 Notes

- PDF is processed into text chunks and embedded using SentenceTransformers.
- FAISS is used for fast similarity search on the document chunks.
- Conversation memory allows the model to retain chat history.
- PDF size is limited to 300KB to keep performance smooth.

---

## 🧪 Example Use Case

> Upload a user manual or policy document and ask things like:
> - _"What is the return policy?"_
> - _"Can you summarize the key features?"_

---

## ❗ Troubleshooting

- **App not starting?** Ensure all dependencies are installed.
- **Error: file too large?** Compress or reduce your PDF under 300KB.
- **Model not responding?** Verify your OpenAI API Key is active and correct.

---

## 📃 License

MIT License

---

## 🤝 Contributing

Pull requests and issues are welcome!
