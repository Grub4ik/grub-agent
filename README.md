# Grub Agent

A local voice assistant for Linux. Grub captures speech from a microphone, transcribes it with `faster-whisper`, sends the text to a local Ollama model, and displays the reply in the terminal.

> A modular, local-first AI voice assistant for Linux, powered by local LLMs, speech recognition, and custom tools.

## v0.1 features

- Push-to-talk: hold F8 to speak and release it to transcribe.
- Terminal fallback: press Enter to start recording and press Enter again to stop.
- `faster-whisper` is loaded once and reused for the lifetime of the process.
- Ollama is called through its local HTTP API; conversation history is bounded by a configurable number of turns.
- Thinking output is disabled in Ollama requests so only the final answer is shown.
- Transcription time, LLM response time, and total processing latency are displayed.
- A failed recording or request does not terminate the assistant loop.

## Requirements

- Linux Mint and Python 3.10 or newer (developed with Python 3.12).
- Ollama and the `gemma4:12b` model.
- A microphone accessible through PortAudio.
- An NVIDIA GPU with compatible CUDA/cuDNN runtime libraries for GPU-based `faster-whisper` inference.
- Internet access for the first Whisper model download. Transcription runs locally after the model is cached.

F8 is handled by `pynput` without root access. It usually works directly on X11. Wayland may restrict global key capture; in that case, Grub falls back to terminal Enter controls.

## Installation

```bash
cd ~/Projects/grub-agent
sudo apt install python3-venv python3-dev portaudio19-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Make sure Ollama is running and the model is available:

```bash
ollama pull gemma4:12b
ollama list
```

Run `ollama pull` only if the model has not already been downloaded. On its first run, `faster-whisper` downloads the `small` model and stores it in its local cache.

## Run

From the project root:

```bash
source .venv/bin/activate
python main.py
```

When Grub reports that it is ready, hold F8, speak, and release the key. Press Ctrl+C to exit. If F8 is unavailable, follow the Enter-mode prompts.

List available input devices with:

```bash
python -c 'import sounddevice as sd; print(sd.query_devices())'
```

Select a device by its listed index:

```bash
GRUB_AUDIO_DEVICE=2 python main.py
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GRUB_MODEL` | `gemma4:12b` | Ollama model name |
| `GRUB_OLLAMA_URL` | `http://localhost:11434` | Local Ollama API URL |
| `GRUB_OLLAMA_TIMEOUT` | `120` | Request timeout in seconds |
| `GRUB_OLLAMA_KEEP_ALIVE` | `10m` | How long Ollama keeps the model loaded after a request |
| `GRUB_HISTORY_TURNS` | `10` | Number of user/assistant message pairs to retain |
| `GRUB_STT_MODEL` | `small` | `faster-whisper` model |
| `GRUB_STT_DEVICE` | `cuda` | `faster-whisper` device: `cuda` or `cpu` |
| `GRUB_STT_COMPUTE_TYPE` | `float16` | `faster-whisper` compute type |
| `GRUB_STT_LANGUAGE` | `ru` | Speech language; set to `auto` for automatic detection |
| `GRUB_AUDIO_DEVICE` | System default | Input device index from sounddevice |

Whisper and Gemma share GPU memory. If loading both causes a CUDA error or Ollama offloads part of the LLM to system memory, run speech recognition on the CPU:

```bash
GRUB_STT_DEVICE=cpu GRUB_STT_COMPUTE_TYPE=int8 python main.py
```

Automatic GPU memory management is not implemented. Moving STT to the CPU may increase transcription time.

## Syntax check

```bash
python -m compileall -q main.py grub
```

## Project structure

```text
grub-agent/
├── grub/
│   ├── __init__.py
│   ├── assistant.py
│   ├── audio.py
│   ├── llm.py
│   └── stt.py
├── .gitignore
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

## Technologies

- Python — application language.
- sounddevice and NumPy — microphone capture and in-memory audio buffers.
- faster-whisper / CTranslate2 — local speech recognition.
- Ollama HTTP API — local LLM chat.
- pynput — F8 push-to-talk when supported by the desktop session.

## Roadmap

TTS, wake word, VAD, streaming STT, GUI, tools and action execution, MCP, vector memory, and autostart are not implemented yet.
