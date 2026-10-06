import asyncio
import io
import json
import os
import secrets
import time
import wave

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

# ---------- Config (Set these as Render environment variables) ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEVICE_TOKEN = os.getenv("DEVICE_TOKEN", "my-secret-token")     # shared secret for ESP and app
GROQ_MODEL   = os.getenv("GROQ_MODEL", "whisper-large-v3-turbo")
LANGUAGE     = os.getenv("LANGUAGE", "")                      # e.g. "en"; empty = auto-detect
WEBHOOK_URL  = os.getenv("WEBHOOK_URL", "")                   # optional: POST every transcript here

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MIN_SECONDS = 0.4                                              # ignore blips shorter than this
MAX_SECONDS = 30                                               # safety cap per utterance
DEFAULT_RATE = 16000                                           # 16 kHz native Whisper rate

# Whisper tends to invent these on near-silence
HALLUCINATIONS = {"thank you.", "thanks for watching!", "you", "bye.", "."}

app = FastAPI(title="ESP32 Voice-to-Text Server")
http = httpx.AsyncClient(timeout=30)
app_clients: set[WebSocket] = set()                            # live app clients listening for transcripts


# ---------- Audio Cleaning & Normalization ----------
def clean_pcm(pcm: bytes) -> bytes:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return pcm
    x -= x.mean()
    loud = np.percentile(np.abs(x), 99.5)
    if loud > 1:
        x *= min(0.8 * 32767 / loud, 20)        # normalise volume, capped gain
    return np.clip(x, -32768, 32767).astype(np.int16).tobytes()


# ---------- Swappable STT (Replace this function for local Raspberry Pi Whisper) ----------
async def transcribe(wav_bytes: bytes) -> str:
    if not GROQ_API_KEY:
        print("[Error] GROQ_API_KEY environment variable is not set!")
        return "Error: GROQ_API_KEY missing"

    data = {"model": GROQ_MODEL, "response_format": "json", "temperature": "0"}
    if LANGUAGE:
        data["language"] = LANGUAGE

    r = await http.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        data=data,
    )
    r.raise_for_status()
    return r.json().get("text", "").strip()


# ---------- Output format ----------
def format_output(text: str, device: str) -> dict:
    return {"type": "transcript", "device": device, "text": text, "ts": int(time.time())}


# ---------- Helpers ----------
def pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)          # 16-bit PCM
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def broadcast(payload: dict):
    dead = []
    for c in app_clients:
        try:
            await c.send_json(payload)
        except Exception:
            dead.append(c)
    for c in dead:
        app_clients.discard(c)


async def post_webhook(payload: dict):
    if not WEBHOOK_URL:
        return
    try:
        await http.post(WEBHOOK_URL, json=payload)
    except Exception as e:
        print("webhook failed:", e)


def token_ok(token: str) -> bool:
    return secrets.compare_digest(token, DEVICE_TOKEN)


async def handle_utterance(ws: WebSocket, device: str, pcm: bytes, rate: int):
    seconds = len(pcm) / 2 / rate
    if seconds < MIN_SECONDS:
        await ws.send_json({"type": "transcript", "device": device, "text": ""})
        return
    try:
        await ws.send_json({"type": "status", "text": "thinking"})
        text = await transcribe(pcm_to_wav(clean_pcm(pcm), rate))
        if text.lower() in HALLUCINATIONS:
            text = ""
        out = format_output(text, device)
        await ws.send_json(out)                       # back to the ESP -> OLED
        if text:
            await broadcast(out)                      # broadcast live to connected app(s)
            await post_webhook(out)                   # POST to external webhook, if set
        print(f"[{device}] {seconds:.1f}s audio -> Transcript: {text!r}")
    except Exception as e:
        print("transcribe error:", e)
        try:
            await ws.send_json({"type": "error", "text": str(e)[:80]})
        except Exception:
            pass


# ---------- Routes ----------
@app.get("/")
@app.get("/health")
async def health():
    return {"ok": True, "service": "ESP32 Voice Server"}


@app.websocket("/ws/esp")
async def esp_socket(ws: WebSocket, token: str = Query(""), device: str = Query("esp1")):
    if not token_ok(token):
        await ws.close(code=1008)
        return
    await ws.accept()
    print(f"[{device}] connected via WebSocket")

    recording = False
    rate = DEFAULT_RATE
    pcm = bytearray()

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                if recording and len(pcm) < MAX_SECONDS * rate * 2:
                    pcm.extend(msg["bytes"])

            elif msg.get("text") is not None:
                try:
                    cmd = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                t = cmd.get("type")
                if t == "start":
                    rate = int(cmd.get("rate", DEFAULT_RATE))
                    pcm = bytearray()
                    recording = True
                elif t == "end" and recording:
                    recording = False
                    data, pcm = bytes(pcm), bytearray()
                    asyncio.create_task(handle_utterance(ws, device, data, rate))
                elif t == "ping":
                    await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        print(f"[{device}] disconnected")


@app.websocket("/ws/app")
async def app_socket(ws: WebSocket, token: str = Query("")):
    """Your app connects here to receive every transcript live."""
    if not token_ok(token):
        await ws.close(code=1008)
        return
    await ws.accept()
    app_clients.add(ws)
    print("App client connected")
    try:
        while True:
            await ws.receive_text()                   # keep-alive; ignore content
    except WebSocketDisconnect:
        pass
    finally:
        app_clients.discard(ws)
        print("App client disconnected")
