# SETUP — Local Voice Agent with RAG

Complete setup guide for running the voice AI agent entirely on your local machine.  
**No cloud APIs, no API keys, no internet required at runtime.**

---

## Prerequisites

| Tool | Purpose | Check |
|------|---------|-------|
| **Python 3.10+** | Agent runtime | `python --version` |
| **Docker Desktop** | Run STT (Speaches) and TTS (Kokoro) servers | `docker --version` |
| **Ollama** | Local LLM + embeddings | `ollama --version` |
| **LiveKit Server** | Real-time media transport | `livekit-server --help` |

### Install Prerequisites

1. **Python 3.10+**: Download from [python.org](https://www.python.org/downloads/)

2. **Docker Desktop**: Download from [docker.com](https://www.docker.com/products/docker-desktop/)
   - After installing, make sure Docker Desktop is **running** (look for the whale icon in your system tray)

3. **Ollama**: Download from [ollama.com](https://ollama.com/download)
   - The installer adds `ollama` to your PATH automatically

4. **LiveKit Server**: Install via WinGet (recommended):
   ```powershell
   winget install LiveKit.LiveKitCLI
   ```
   Or download the binary manually from [GitHub releases](https://github.com/livekit/livekit/releases) and add it to your PATH.

---

## Step 1 — Pull Ollama Models

Open a terminal and pull the two models we need:

```powershell
# LLM for answering questions (~2GB download)
ollama pull llama3.2:3b

# Embedding model for RAG retrieval (~270MB download)
ollama pull nomic-embed-text
```

Verify they're available:

```powershell
ollama list
# Should show both llama3.2:3b and nomic-embed-text
```

---

## Step 2 — Start External Services

You need **4 services** running, each in its own terminal window. Start them in this order:

### Terminal 1 — LiveKit Dev Server

```powershell
livekit-server --dev
```

You should see output like:
```
LiveKit server started
  API Key:    devkey
  API Secret: secret
  WebSocket:  ws://localhost:7880
```

> **Note:** Keep this terminal open. The server runs in the foreground.

### Terminal 2 — Ollama

Ollama may already be running as a background service. Check with:

```powershell
curl http://localhost:11434/api/tags
```

If you get a response, Ollama is already running. If not, start it:

```powershell
ollama serve
```

### Terminal 3 — Kokoro TTS (Text-to-Speech)

```powershell
docker run -d -p 8880:8880 --name kokoro-tts ghcr.io/remsky/kokoro-fastapi-cpu:latest
```

Wait ~30 seconds for the model to load, then verify:

```powershell
curl http://localhost:8880/v1/audio/voices
```

You should see a JSON list of available voices (we use `af_heart`).

> **GPU variant** (optional, uses VRAM): Replace `kokoro-fastapi-cpu` with `kokoro-fastapi-gpu` and add `--gpus all`.

### Terminal 4 — Speaches STT (Speech-to-Text)

```powershell
docker run -d -p 8300:8000 --name speaches-stt ghcr.io/speaches-ai/speaches:latest
```

Wait ~30 seconds for first-time model download, then verify:

```powershell
curl http://localhost:8300/v1/models
```

> **Note:** The first transcription request will trigger the Whisper model download (~500MB). Subsequent requests are instant.

---

## Step 3 — Python Environment Setup

Open a **new terminal** and navigate to the project:

```powershell
cd "D:\Internships\NESCOM Intership - Project Askari Bank\rag-project"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Step 4 — Test RAG Standalone (Optional but Recommended)

Before running the full voice agent, test that the RAG pipeline works on its own:

```powershell
python rag.py
```

This will:
1. Load `data/knowledge.pdf`
2. Extract and chunk the text
3. Embed all chunks using Ollama's `nomic-embed-text`
4. Start an interactive Q&A loop

Type questions about your PDF content and verify the answers are grounded and accurate. Press `Ctrl+C` to exit.

---

## Step 5 — Run the Voice Agent

With all 4 services running (LiveKit, Ollama, Kokoro, Speaches), launch the agent in **console mode**:

```powershell
python agent.py console
```

This starts a terminal-based voice session:
- **Speak into your microphone** — your speech is transcribed in real-time
- **The agent retrieves relevant context** from your PDF
- **Ollama generates a grounded answer**
- **The answer is spoken back** through your speakers

> **Tip:** Press `Ctrl+T` in the terminal to toggle text mode (type instead of speak). Useful for quick testing.

---

## Quick Health Check Commands

Run these to verify all services are up before starting the agent:

```powershell
# LiveKit dev server
curl http://localhost:7880

# Ollama
curl http://localhost:11434/api/tags

# Kokoro TTS
curl http://localhost:8880/v1/audio/voices

# Speaches STT
curl http://localhost:8300/v1/models
```

---

## Stopping Services

```powershell
# Stop Docker containers
docker stop kokoro-tts speaches-stt
docker rm kokoro-tts speaches-stt

# LiveKit and Ollama: just close their terminal windows (or Ctrl+C)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot connect to Ollama` | Run `ollama serve` or check if it's already running |
| `Docker: port already in use` | Stop the existing container: `docker stop <name>` then `docker rm <name>` |
| `No audio detected` | Check your system microphone permissions and default input device |
| `Slow first response` | Normal — Speaches downloads the Whisper model on first use (~500MB) |
| `LLM gives wrong answers` | Test RAG standalone first (`python rag.py`) to check retrieval quality |
| `VRAM out of memory` | The STT and TTS run on CPU by default. Only Ollama uses the GPU. If still OOM, try `ollama run llama3.2:1b` (smaller model) |

---

## Architecture Reference

```
rag-project/
├── data/
│   └── knowledge.pdf        # Your PDF knowledge base
├── rag.py                    # Standalone RAG pipeline (PDF → chunks → embeddings → retrieval)
├── agent.py                  # LiveKit voice agent (STT → RAG → LLM → TTS)
├── .env.local                # LiveKit dev server credentials
├── requirements.txt          # Python dependencies
└── SETUP.md                  # This file
```

**Services diagram:**
```
┌─────────────────────────────────────────────────────────────┐
│  Your Terminal (python agent.py console)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Silero   │→ │ Speaches │→ │ RAG +    │→ │ Kokoro     │ │
│  │ VAD      │  │ STT      │  │ Ollama   │  │ TTS        │ │
│  │ (in-proc)│  │ :8300    │  │ :11434   │  │ :8880      │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘ │
│       ↑                                          ↓         │
│   🎤 Mic                                    🔊 Speakers    │
└─────────────────────────────────────────────────────────────┘
│  LiveKit Server (:7880) — media transport layer             │
└─────────────────────────────────────────────────────────────┘
```
