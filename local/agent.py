"""
agent.py — LiveKit Voice Agent with RAG (No-Docker Edition)
============================================================
A terminal-based voice AI agent using 100% local, no-cloud components:
  - VAD:  Silero (in-process, bundled with livekit-plugins-silero)
  - STT:  faster-whisper (in-process, CPU, via FasterWhisperSTT below)
  - LLM:  Ollama llama3.2:3b (separate process, already running)
  - TTS:  Windows SAPI via pyttsx3 (in-process, Pyttsx3TTS below)
  - RAG:  rag.py (in-process, numpy cosine similarity)

Run with:
    python agent.py console
"""

import asyncio
import io
import logging
import os
import tempfile
import time
import uuid
import wave

import numpy as np
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, JobContext, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents import stt, tts, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import openai, silero

from rag import load_and_chunk, embed_texts, retrieve, build_rag_prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(__file__), ".env.local"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "knowledge.pdf")

logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)


# ===========================================================================
# Custom STT: faster-whisper (in-process, CPU)
# ===========================================================================

class FasterWhisperSTT(stt.STT):
    """
    Wraps the faster-whisper library as a LiveKit STT plugin.

    Runs Whisper on CPU — keeps the GPU free for Ollama.
    Model is downloaded once from Hugging Face on first use (~150 MB for 'small').
    """

    def __init__(self, model_size: str = "small"):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False),
        )
        self._model_size = model_size
        self._model = None  # Lazy-loaded on first recognition

    def _load_model(self):
        """Load the Whisper model (blocking — called in a thread)."""
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel
        logger.info(f"[STT] Loading faster-whisper model '{self._model_size}' on CPU...")
        # int8 quantization: minimal VRAM/RAM, fast on CPU
        self._model = WhisperModel(
            self._model_size,
            device="cpu",
            compute_type="int8",
        )
        logger.info("[STT] faster-whisper model loaded.")
        return self._model

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options,
    ) -> stt.SpeechEvent:
        """
        Convert a complete audio buffer (one utterance) to text.
        Called by LiveKit after Silero VAD marks the end of a speech segment.
        """
        # Merge all frames into a single frame
        merged = utils.merge_frames(buffer)

        # Convert 16-bit PCM → float32 in range [-1, 1]
        audio_array = (
            np.frombuffer(merged.data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # If audio has multiple channels, take the first
        if merged.num_channels > 1:
            audio_array = audio_array.reshape(-1, merged.num_channels)[:, 0]

        # Resample to 16 kHz if necessary (faster-whisper expects 16 kHz)
        src_rate = merged.sample_rate
        if src_rate != 16000:
            try:
                from scipy.signal import resample_poly
                import math
                g = math.gcd(16000, src_rate)
                audio_array = resample_poly(audio_array, 16000 // g, src_rate // g)
            except ImportError:
                # scipy not available — warn and proceed (may affect accuracy)
                logger.warning("[STT] scipy not found; audio not resampled from %d Hz", src_rate)

        # Run Whisper transcription in a thread (blocking C++ call)
        def _transcribe():
            model = self._load_model()
            lang = language if language is not NOT_GIVEN else "en"
            segments, info = model.transcribe(
                audio_array,
                beam_size=5,
                language=lang,
                vad_filter=False,  # Silero VAD already handles this
            )
            text = " ".join(seg.text for seg in segments).strip()
            return text, info.language

        text, detected_lang = await asyncio.to_thread(_transcribe)
        logger.info(f"[STT] Transcribed: '{text}'")

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=detected_lang)],
        )

    # stream() intentionally not overridden — the AgentSession uses
    # the VAD we pass at session creation time, not one inside STT.


# ===========================================================================
# Custom TTS: pyttsx3 / Windows SAPI (in-process)
# ===========================================================================

class Pyttsx3ChunkedStream(tts.ChunkedStream):
    """
    Synthesizes speech using pyttsx3 (Windows SAPI), saves it to a
    temporary WAV file, reads the raw PCM bytes, and feeds them to
    LiveKit's audio pipeline via AudioEmitter.
    """

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Generate WAV audio from text, then push PCM bytes to LiveKit."""
        import pyttsx3

        # --- Synthesize to a temporary WAV file (blocking) ---
        def _synthesize() -> bytes:
            tmp_path = tempfile.mktemp(suffix=".wav")
            engine = pyttsx3.init()
            # Set a comfortable speaking rate (words per minute)
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 1.0)
            engine.save_to_file(self._input_text, tmp_path)
            engine.runAndWait()
            engine.stop()
            with open(tmp_path, "rb") as f:
                data = f.read()
            os.unlink(tmp_path)
            return data

        wav_bytes = await asyncio.to_thread(_synthesize)

        # --- Parse WAV header to get actual sample rate / channels ---
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            # sample_width = wf.getsampwidth()  # should be 2 (16-bit)
            pcm_data = wf.readframes(wf.getnframes())  # raw 16-bit PCM

        # --- Initialize the emitter with the actual audio format ---
        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=sample_rate,
            num_channels=num_channels,
            mime_type="audio/pcm",   # raw 16-bit LE PCM
            frame_size_ms=200,
        )

        # --- Push PCM data ---
        output_emitter.push(pcm_data)


class Pyttsx3TTS(tts.TTS):
    """
    LiveKit TTS plugin backed by pyttsx3 (Windows SAPI).

    Completely local — no API key, no internet, no Docker.
    Voice quality: Windows built-in TTS (David/Zira on most systems).
    """

    def __init__(self):
        # 22050 Hz is typical for Windows SAPI; we read the actual rate from
        # the WAV header in ChunkedStream._run() and pass it to the emitter.
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=22050,
            num_channels=1,
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options=None,
    ) -> Pyttsx3ChunkedStream:
        opts = conn_options if conn_options is not None else DEFAULT_API_CONNECT_OPTIONS
        return Pyttsx3ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=opts,
        )


# ===========================================================================
# RAG-enhanced Agent
# ===========================================================================

class RAGVoiceAgent(Agent):
    """
    Subclasses LiveKit's Agent to inject RAG context before every LLM response.

    `on_user_turn_completed` fires after STT finishes but before the LLM is called.
    We use it to always retrieve relevant PDF chunks — no tool calls needed.
    """

    def __init__(self, chunks: list[str], chunk_vectors: np.ndarray, **kwargs):
        super().__init__(
            instructions=(
                "You are the Askari Bank HR Policy Assistant — a friendly, warm, and knowledgeable guide "
                "dedicated to helping Askari Bank employees understand HR policies, benefits, leave rules, "
                "code of conduct, and all related topics. "
                "Always speak in a helpful, cheerful, and encouraging tone — like a trusted colleague who genuinely wants to help. "
                "Use the retrieved HR policy context as your primary source of information. "
                "If the context partially covers a question, enrich it with your broader knowledge of HR best practices to give the most complete, useful answer. "
                "If a question is completely unrelated to HR or Askari Bank, gently let the employee know your specialisation with a friendly suggestion. "
                "Never give blunt refusals. Always wrap any limitation in warmth and offer to help with related topics. "
                "Do NOT start your response with the word 'Assistant' or any role label — just speak naturally and directly. "
                "Keep answers concise and conversational, as they will be spoken aloud."
            ),
            **kwargs,
        )
        self._chunks = chunks
        self._chunk_vectors = chunk_vectors

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """
        Runs retrieval on every user turn and injects the top-3 PDF chunks
        as a system message so the LLM answers are grounded in the document.
        """
        user_text = new_message.text_content
        if not user_text or not user_text.strip():
            return

        logger.info(f"[RAG] Retrieving context for: '{user_text}'")

        # Embed query and search (blocking I/O → run in thread)
        retrieved = await asyncio.to_thread(
            retrieve,
            user_text,
            self._chunks,
            self._chunk_vectors,
        )

        for i, (chunk, score) in enumerate(retrieved, 1):
            logger.info(f"[RAG]   #{i} score={score:.3f}: {chunk[:60].replace(chr(10),' ')}...")

        # Build the RAG system prompt and inject it into the conversation
        rag_prompt = build_rag_prompt(user_text, retrieved)
        turn_ctx.add_message(role="system", content=rag_prompt)
        logger.info("[RAG] Context injected.")


# ===========================================================================
# Entrypoint
# ===========================================================================

async def entrypoint(ctx: JobContext):
    """
    Called once per session by the LiveKit Agents worker.
    Loads the knowledge base, wires all components, starts the voice loop.
    """
    await ctx.connect()

    # --- Load and embed the PDF knowledge base (once at startup) ---
    logger.info(f"[STARTUP] Loading: {PDF_PATH}")
    chunks = load_and_chunk(PDF_PATH)
    if not chunks:
        logger.error("[STARTUP] No text extracted from PDF!")
        return

    logger.info(f"[STARTUP] Embedding {len(chunks)} chunks...")
    t0 = time.time()
    chunk_vectors = await asyncio.to_thread(embed_texts, chunks)
    logger.info(f"[STARTUP] Embeddings ready in {time.time()-t0:.1f}s — shape: {chunk_vectors.shape}")

    # --- Build components ---
    stt_plugin = FasterWhisperSTT(model_size="small")

    # Ollama LLM via its OpenAI-compatible endpoint (no API key needed)
    llm_plugin = openai.LLM.with_ollama(
        model=CHAT_MODEL,
        base_url=f"{OLLAMA_BASE_URL}/v1",
    )

    tts_plugin = Pyttsx3TTS()

    # Silero VAD — in-process, no cloud, just detects voice activity
    vad_plugin = silero.VAD.load(
        min_speech_duration=0.2,       # Require longer speech duration to trigger (ignore brief noises)
        min_silence_duration=1.0,      # Wait longer before considering speech finished
        activation_threshold=0.8,      # Require higher confidence (ignore background noise)
    )

    # --- Start the voice session ---
    session = AgentSession(
        stt=stt_plugin,
        llm=llm_plugin,
        tts=tts_plugin,
        vad=vad_plugin,
    )

    agent = RAGVoiceAgent(
        chunks=chunks,
        chunk_vectors=chunk_vectors,
    )

    await session.start(room=ctx.room, agent=agent)

    # Greet the user once the session is live
    await session.generate_reply(
        instructions=(
            "Introduce yourself warmly and enthusiastically as the Askari Bank HR Policy Assistant. "
            "Welcome the employee and let them know you're here to help with any HR policy questions — "
            "whether it's about leave, benefits, code of conduct, or anything else. "
            "Keep it friendly, upbeat, and brief (2-3 sentences). "
            "Do NOT begin your response with the word 'Assistant' or any label — speak directly and naturally."
        )
    )

    logger.info("[STARTUP] Voice agent is live — speak into your microphone.")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    # python agent.py console  →  terminal voice session (mic + speakers)
    # python agent.py start    →  connect to LiveKit server as a worker
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint),
    )
