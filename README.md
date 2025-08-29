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

## 📂 Project Structure
ritu29verma-semantic-engine/
│── app.py # Main Streamlit app
│── flashcard_generator.py # Flashcard generation logic
│── pdf_utils.py # PDF parsing utilities
│── requirements.txt # Dependencies list

yaml
Copy code

---

## ⚡ Setup & Installation (All Steps)

Follow these steps to set up this project on any system:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ritu29verma-semantic-engine.git
cd ritu29verma-semantic-engine
2️⃣ Create Virtual Environment
bash
Copy code
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On Linux / Mac
python3 -m venv .venv
source .venv/bin/activate
3️⃣ Install Dependencies
bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
4️⃣ Run the App
bash
Copy code
streamlit run app.py
👉 Then open the app in your browser at:
http://localhost:8501

📖 Example Workflow
Upload a PDF document (book, paper, notes).

Use semantic search to find key answers.

Select a page range and generate flashcards.

Revise and learn using the generated flashcards.

🛠️ Tech Stack
Python 3.9+

Streamlit

Transformers

Sentence-Transformers

PyPDF2 / pdfplumber

🤝 Contributing
Contributions are welcome!

Fork the repo

Create a new branch

Submit a Pull Request

📜 License
Licensed under the MIT License.

👩‍💻 Author
Ritu Verma
✨ Building tools for smarter learning & semantic understanding




Cha
