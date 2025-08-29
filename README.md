# 📘 Semantic Search & Flashcard Generator

A **Semantic Engine** built with **Streamlit** that lets you:
- Upload and parse PDF documents
- Perform **semantic search** over content
- Generate **flashcards** from selected text ranges
- Summarize long texts for better learning

---

## 🚀 Features
- 📂 **PDF Upload & Parsing** – Extract text from PDFs  
- 🔍 **Semantic Search** – Find contextually relevant answers  
- 📝 **Flashcard Generator** – Create Q&A style flashcards for revision  
- 📖 **Book Summarization** – Summarize content from large documents  

---

## ⚡ Setup & Installation (All Steps)

Follow these steps to set up this project on any system:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ritu29verma-semantic-engine.git
cd ritu29verma-semantic-engine

```

2️⃣ Create Virtual Environment

# On Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

# On Linux / Mac
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4️⃣ Run the App

```bash
streamlit run app.py
```

👉 Then open the app in your browser at:

```bash
http://localhost:8501
```

📖 Example Workflow
Upload a PDF document (book, paper, notes).

Use semantic search to find key answers.

Select a page range and generate flashcards.

Revise and learn using the generated flashcards.

## 🛠️ Tech Stack (Detailed)

This project is built with a clean **FE–BE separation** inside a single Streamlit app.

---

### 🌐 Frontend
- **Framework**: [Streamlit 1.34+](https://streamlit.io/)  
  - Used for building the interactive UI (file upload, search bar, flashcard display).  
  - Custom **CSS styling** for flashcards with flip animations.  
  - Columns and layouts to organize results side-by-side.  

---

### ⚙️ Backend (App Logic)
- **Language**: Python **3.9+**  
- **Core App File**: `app.py`  
  - Handles routing within Streamlit (upload → parse → search → flashcards).  
- **Modules**:  
  - `flashcard_generator.py` → Generates Q&A pairs using NLP models.  
  - `pdf_utils.py` → PDF parsing, cleaning, and preprocessing.  

---

### 🤖 NLP / ML Models
- **[Transformers](https://huggingface.co/transformers/)** (v4.40+)  
  - For text summarization and generation.  
  - Example: `"facebook/bart-large-cnn"` for summarization.  

- **[Sentence-Transformers](https://www.sbert.net/)** (v2.6+)  
  - Semantic embeddings (vector representation of text).  
  - Example: `"all-MiniLM-L6-v2"` for efficient semantic search.  

- **[Hugging Face Hub](https://huggingface.co/)**  
  - Pre-trained models are downloaded dynamically.  

---

### 📂 Data Handling
- **PDF Extraction**:  
  - [PyPDF2](https://pypi.org/project/PyPDF2/) (basic parsing)  
  - OR [pdfplumber](https://pypi.org/project/pdfplumber/) (more accurate text + layout parsing)  

- **Vector Search**:  
  - Embeddings stored temporarily in-memory for semantic similarity.  
  - Uses **cosine similarity** for ranking answers.  

- **Flashcards Data**:  
  - Stored as Python dict objects in-session.  
  - Not persisted in DB (lightweight demo version).  

---

### 🗄️ Storage (Current)
- Local session state in **Streamlit**.  
- No external database required.  
- (🔜 Can be extended with **FAISS / Pinecone / Weaviate** for scalable vector storage).  

---

### 📦 Dependencies & Versions
- Python ≥ 3.9  
- streamlit ≥ 1.34  
- transformers ≥ 4.40  
- sentence-transformers ≥ 2.6  
- PyPDF2 ≥ 3.0 / pdfplumber ≥ 0.11  
- torch (PyTorch) ≥ 2.2 (required for models)  

---

### ☁️ Deployment Options
- **Localhost**: Run with `streamlit run app.py`  
- **Cloud Deployment**:  
  - [Streamlit Cloud](https://streamlit.io/cloud)  
  - [Heroku](https://www.heroku.com/)  
  - [Railway](https://railway.app/)  
  - [Docker](https://www.docker.com/) container for reproducible setup  

---

### 🔮 Future Enhancements
- Persistent database for storing flashcards.  
- Multi-user authentication.  
- Scalable vector DB (Pinecone, Weaviate, FAISS).  
- Support for **images + tables** in PDFs.  
- Export flashcards to **Anki** or CSV.  


🤝 Contributing
Contributions are welcome!

```bash
Fork the repo

Create a new branch

Submit a Pull Request
```
📜 License
Licensed under the MIT License.

👩‍💻 Author
Ritu Verma
✨ Building tools for smarter learning & semantic understanding
