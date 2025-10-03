# 🧠 One-Shot: Last Minute Exam Preparation Platform with Explainable AI (XAI)

---

## 📌 Project Overview
Engineering students often struggle with **last-minute exam preparation** due to scattered resources and lack of structured revision.  
**One-Shot** addresses this challenge by transforming lecture slides and past exam papers into:  
- Concise summaries  
- AI-generated practice questions  
- Adaptive mock tests  
- Explainable AI (XAI) insights highlighting likely exam questions  
- A video-enabled peer learning workspace  

This ensures students can prepare **faster, smarter, and more effectively**.

---

## 🎯 Objectives
- Deliver a **productive last-minute exam-preparation tool** tailored to engineering curricula.  
- Analyze lecture slides and past-year papers with NLP to surface high-priority topics.  
- Generate **AI-driven mock tests** weighted by topic importance, frequency, and historical patterns.  
- Provide **explainable AI insights** that justify likely exam questions.  
- Enable **real-time collaboration** with video chat, whiteboards, and screen sharing.  
- Offer personalized recommendations through **analytics & dashboards**.  
- Integrate an **AI chatbot** for on-demand clarification.

---

## ⚙️ Tech Stack
- **Programming Language:** Python  
- **Libraries:** LangChain, Pinecone, Scikit-learn, NLTK, Streamlit  
- **Environment:** VS Code  
- **Tools:** NumPy, PyPDF  
- **Frontend:** React.js, Tailwind CSS  
- **Backend:** Django  
- **Deployment:** Docker + AWS 

---

## 📂 Project Structure
```

One-Shot-Implementation/
│── .gitignore
│── .env                   # API keys & tokens (not committed)
│── requirements.txt       # Dependencies
│── main.py                # Entry point for Basic XAI App
│── config.py
│── document_processor.py
│── llmembedding_setup.py
│── pinecone_setup.py
│── prompts.py
│── ui_components.py
│
└── xai/                   # Advanced XAI App
    │── main.py
    │── config.py
    │── document_processor.py
    │── llmembedding_setup.py
    │── pinecone_setup.py
    │── prompts.py
    │── summarai_utils.py
    │── ui_components.py

````

---

## 🚀 Setup & Implementation

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
````

### 2️⃣ Activate Virtual Environment

* **Windows (PowerShell)**

  ```bash
  venv\Scripts\activate
  ```
* **Linux / macOS**

  ```bash
  source venv/bin/activate
  ```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables

Create a **`.env`** file in the project root with:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENV=your_pinecone_environment_here  # e.g., us-east-1
```

### 5️⃣ Run Applications

**Basic XAI App**

```bash
streamlit run main.py
```

**Advanced XAI App**

```bash
cd xai
streamlit run main.py
```

---

## 🙏 Acknowledgements

* Hugging Face
* Pinecone
* Groq AI
* Streamlit
* Research papers on **XAI in Education**

---
