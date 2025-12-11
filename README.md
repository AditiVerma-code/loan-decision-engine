# 🚀 **StrategyHub AI — Intelligent Trading Strategy Generator**

**React (Vite) • FastAPI • LLM (Gemini / Ollama) • Backtrader • Python**

StrategyHub AI is a full-stack application that transforms natural-language trading ideas into executable Python trading strategies. The system is engineered with **production-grade architecture**, including LLM orchestration, local fallbacks, and safe strategy generation workflows.

This project demonstrates strong skills in **AI engineering, backend development, trading logic, and full-stack system design**.

---

## ⭐ **Key Features**

### 🔹 Natural Language → Trading Strategy Code

Users can describe a strategy, such as:

> “Create an RSI strategy for MSFT, buy below 30 and sell above 70.”

StrategyHub AI converts this into a complete **Backtrader** strategy class ready for execution.

---

### 🔹 Robust Multi-Layer LLM Pipeline

LLM requests follow a resilient fallback chain:

1. **Primary LLM:** Google Gemini
2. **Fallback:** Local Ollama model (e.g., Mistral, Llama)
3. **Guaranteed Output:** Demo fallback text if LLMs are unavailable

This ensures the backend **never hangs** and always returns a structured response.

---

### 🔹 Modern, Modular Backend (FastAPI)

* Async execution with threadpool safety
* CORS-enabled for frontend communication
* Clear separation of LLM logic and strategy parsing
* Environment-secured API key handling

---

### 🔹 Clean React (Vite) Frontend

* User-friendly prompt input
* Displays generated trading strategy code
* Easily extendable for charts, backtesting, or UI enhancements

---

## 🧱 **Architecture Overview**

```
┌────────────────┐        ┌────────────────────────────────┐
│   Frontend     │        │            Backend              │
│ React + Vite   │ -----> │        FastAPI Server           │
└────────────────┘        │  /generate-strategy endpoint    │
                          ├──────────────────────────────────┤
                          │ 1. Gemini (primary LLM)          │
                          │ 2. Ollama local model fallback    │
                          │ 3. Demo response (guaranteed)     │
                          └──────────────────────────────────┘

                ┌──────────────────────────────┐
                │ Backtrader Strategy Builder   │
                │ Converts generated code into  │
                │ executable trading strategies │
                └──────────────────────────────┘
```

---

## 🛠️ **Tech Stack**

### **Frontend**

* React + Vite
* TypeScript
* Modular components

### **Backend**

* FastAPI
* Python
* Backtrader
* Ollama (optional)
* Google Generative AI SDK
* python-dotenv

### **Supporting Tools**

* Uvicorn
* yfinance
* Requests

---

## 🚀 **Getting Started**

### **1. Backend Setup**

```bash
cd backend
pip install -r requirements.txt
pip install python-dotenv requests
```

Create a `.env` file in the backend folder:

```
GEMINI_API_KEY=your_api_key_here
```

Run the server:

```bash
python -m uvicorn server:app --reload
```

Interactive API:

```
http://127.0.0.1:8000/docs
```

---

### **2. Frontend Setup**

```bash
cd frontend
npm install
npm run dev
```

Open in browser:

```
http://localhost:5173/
```

---

## 📡 **API Endpoint**

### `POST /generate-strategy`

**Request Body**

```json
{
  "user_input": "Create an RSI strategy for MSFT"
}
```

**Response**

```json
{
  "strategy_code": "<Generated Python strategy code or fallback>"
}
```

---

## 🔒 **Security**

* API keys managed using `.env`
* `.gitignore` ensures environment files are not committed
* Backend sanitizes errors to avoid exposing internal stack traces

---

## 🧩 **Why This Project Is Valuable**

StrategyHub AI highlights capabilities in:

* Full-stack development
* LLM system integration
* Trading & financial computation
* Backend architecture
* Robust error-handled workflows
* Model fallback strategies (important for real-world ML systems)

This mirrors the expectations of engineering teams at modern fintech, quant, and product-driven companies.

---

## 🔮 **Future Enhancements**

* UI visualization for backtests
* Equity curve & indicator charts
* Multi-model routing (Gemini / GPT / Llama)
* Docker deployment
* User strategy history database
