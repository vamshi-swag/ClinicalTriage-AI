# ⚕️ ClinicalTriage-AI Assistant

A modern, clinical-grade triage assistant built with **Streamlit** and featuring a hybrid architecture of a local **text-retrieval RAG engine** and a **quantized 1B LLM (Llama-3.2-1B-Instruct via Unsloth)**. 

The application is engineered with a **3-Tier Execution Architecture** that guarantees successful deployment on high-end GPU workstations, standard CPU computers, and low-resource cloud containers (like Streamlit Community Cloud).

---

## 🚀 Features

- **Clinical Grounding (RAG)**: Keyword-based lookup in a verified medical guidelines database to verify facts (appendicitis, cholangiocarcinoma, common cold, etc.).
- **3-Tier Intelligent Fallback**:
  - **Tier 1 (GPU/Unsloth)**: Uses native hardware-accelerated 4-bit loading for sub-second responses.
  - **Tier 2 (HF Transformers)**: Runs standard Llama model on standard CUDA or CPU if Unsloth is missing.
  - **Tier 3 (Failsafe Triage)**: Deterministic diagnostic compiler that runs instantly on low-spec hosting servers without requiring heavy model downloads or crashing memory limits.
- **Premium User Experience**: Styled using clean dark clinical themes, Outfit/Inter typography, responsive badges, and card views.

---

## 💻 Local Setup & Execution

### Option A: Via NPM (Node.js wrapper)
If you have Node installed, launch the app directly:
```bash
npm install
npm start
```

### Option B: Via Python Native
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deploying Live to Streamlit Community Cloud (Free)

Streamlit Community Cloud allows you to host Streamlit applications directly from your GitHub repository for free.

### Step 1: Initialize Git and Push to GitHub
1. Open terminal in the project directory.
2. Initialize git and make your first commit:
   ```bash
   git init
   git add .
   git commit -m "initial commit: clinical-triage application"
   ```
3. Create a **New Repository** on [GitHub](https://github.com/new). Name it `ClinicalTriage-AI` (or any name you prefer). Keep it Public or Private.
4. Copy the remote URL from GitHub and push your code:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/ClinicalTriage-AI.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Community Cloud
1. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and click **Sign up / Sign in**.
2. Connect your GitHub account.
3. Click the **Deploy** button (or **Create App**).
4. Fill in the repository details:
   - **Repository**: `YOUR_GITHUB_USERNAME/ClinicalTriage-AI`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy!**

Your application will be live on a custom URL (e.g., `https://clinicaltriage-ai.streamlit.app/`) in a few minutes!
