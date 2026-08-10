# Askari Bank HR AI Call Agent

Welcome to the **Askari Bank HR AI Call Agent** project, developed during the **Ascend Internship**. This project implements an intelligent, voice-based AI assistant designed to help Askari Bank employees navigate HR policies, benefits, leave rules, and codes of conduct naturally through conversation.

The project features a **Retrieval-Augmented Generation (RAG)** pipeline to ground the AI's responses in official Askari Bank HR documentation (provided as a PDF).

This repository contains two variations of the agent:
1. **`local/`**: A 100% local, on-premise version that runs entirely on your machine without relying on external cloud APIs.
2. **`hosted/`**: A cloud-connected version designed to be deployed using LiveKit Cloud.

---

## 📁 Project Structure

### 1. Local Version (`/local`)
The local version is designed for complete data privacy and offline capability. It runs all AI models locally on your hardware.

**Tech Stack:**
- **Framework**: LiveKit Agents for Python
- **VAD (Voice Activity Detection)**: Silero (in-process)
- **STT (Speech-to-Text)**: faster-whisper (runs on CPU via `speaches`)
- **LLM (Large Language Model)**: Ollama (`llama3.2:3b`)
- **Embeddings (For RAG)**: Ollama (`nomic-embed-text`)
- **TTS (Text-to-Speech)**: pyttsx3 (Windows SAPI) or Kokoro TTS
- **RAG Pipeline**: Custom implementation using `pypdf`, `numpy` (cosine similarity), and HTTP requests.

> **To run the local version**, navigate to the `local/` directory and see the detailed [Local Setup Guide](local/SETUP.md).

### 2. Hosted Version (`/hosted`)
The hosted version utilizes LiveKit Cloud for media transport and is optimized for production deployment, enabling integrations with telephony (SIP) or frontend web/mobile apps.

**Tech Stack:**
- **Framework**: LiveKit Agents for Python
- **Media Transport**: LiveKit Cloud
- **Dependency Management**: `uv`

> **To run the hosted version**, navigate to the `hosted/` directory and follow the [Hosted Setup Guide](hosted/README.md).

---

## 🚀 Key Features

- **Conversational Voice Interface**: Speak directly to the agent in real-time.
- **RAG-Powered Knowledge Base**: Answers are accurately grounded in the Askari Bank HR Policy PDF.
- **Local AI Execution**: Optional 100% offline execution for sensitive HR data.
- **Low Latency**: Optimized inference pipelines for smooth, natural voice conversations.

## 🛠️ Quick Start (Local Edition)

If you want to quickly test the agent locally:

1. Ensure you have **Python 3.10+**, **Ollama**, and **LiveKit Server** installed.
2. Pull the required models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
3. Navigate to the local directory and install dependencies:
   ```bash
   cd local
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Follow the steps in `local/SETUP.md` to start the required services and launch the terminal voice session.

---

*Developed by Hashir Ehtisham for the Ascend Internship Program.*
