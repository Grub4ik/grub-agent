# Grub Agent

A local voice assistant for Linux. Grub captures speech from a microphone, transcribes it with `faster-whisper`, sends the text to a local Ollama model, and displays the reply in the terminal and a system notification.

> A modular, local-first AI voice assistant for Linux, powered by local LLMs, speech recognition, and custom tools.

## v0.2 features

- Global X11 push-to-talk: hold F8 in any application to record; release it to transcribe.
- A short rising tone plays before capture and a distinct falling tone plays after capture.
- The start tone finishes before the microphone opens. Speak after hearing the tone.
- A system notification shows the model's reply; the complete reply remains in the terminal.
- Terminal fallback remains available if global keyboard capture is unavailable.
- `faster-whisper` is loaded once and reused for the lifetime of the process.
- Ollama is called through its local HTTP API; conversation history is bounded by a configurable number of turns.
- Thinking output is disabled in Ollama requests so only the final answer is shown.
- Transcription time, LLM response time, and total processing latency are displayed.
- F8 callbacks only enqueue events. Recording coordination runs on the assistant thread, while STT and LLM work run in a worker thread.
- A failed recording, request, sound cue, or notification does not terminate the assistant loop.

## Requirements

- Linux Mint and Python 3.10 or newer (developed with Python 3.12).
- Ollama and the `gemma4:12b` model.
- A microphone accessible through PortAudio.
- An NVIDIA GPU with compatible CUDA/cuDNN runtime libraries for GPU-based `faster-whisper` inference.
- Internet access for the first Whisper model download. Transcription runs locally after the model is cached.

F8 is handled globally by `pynput` without root access. Grub targets X11; Wayland may restrict global key capture. If the listener cannot start or exits, Grub prints the error and falls back to terminal Enter controls.

## Installation

```bash
cd ~/Projects/grub-agent
sudo apt install python3-venv python3-dev portaudio19-dev libnotify-bin
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

Run `main.py` from the project root using the Python environment where the requirements are installed. If you use the existing VoiceSTT environment, activate it first with `source ~/Projects/VoiceSTT/.venv/bin/activate`. Do not run `grub/assistant.py` directly; it is a package module, not an entry point. Calling `/usr/bin/python3` explicitly also bypasses an active virtual environment.

When Grub reports that it is ready, you can minimize the terminal and use other applications. Hold F8, wait for the start tone, speak, and release the key. The full answer appears in the terminal and a system notification shows a shortened version. Press Ctrl+C in the terminal to exit. If F8 is unavailable, follow the Enter-mode prompts.

The start tone is played before audio capture so it is not sent to Whisper. Start speaking after the tone. Sound output uses the system's default playback device independently of the selected microphone.

### Manual X11 check

1. Start Grub from a terminal in the X11 session and wait for the F8-ready message.
2. Minimize the terminal and open a browser or editor.
3. Hold F8, wait for the rising tone, say a short question, and release F8.
4. Confirm the falling tone plays, the full answer appears in the terminal, and a Grub notification appears.

The automated tests use stubs. They do not simulate a physical key press, microphone speech, or a desktop notification.

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
| `GRUB_BEEP_VOLUME` | `0.12` | Feedback cue volume from `0.0` (muted) to `1.0` |
| `GRUB_FEEDBACK_DEVICE` | System default output | Optional sounddevice output device index |

Whisper and Gemma share GPU memory. If loading both causes a CUDA error or Ollama offloads part of the LLM to system memory, run speech recognition on the CPU:

```bash
GRUB_STT_DEVICE=cpu GRUB_STT_COMPUTE_TYPE=int8 python main.py
```

Automatic GPU memory management is not implemented. Moving STT to the CPU may increase transcription time.

## Syntax check

```bash
python -m compileall -q main.py grub
python -m unittest discover -s tests -v
```

The unit tests use stubs for keyboard callbacks, Ollama, notifications, and audio playback. They do not test physical devices.

## Project structure

```text
grub-agent/
├── grub/
│   ├── __init__.py
│   ├── assistant.py
│   ├── audio.py
│   ├── feedback.py
│   ├── llm.py
│   └── stt.py
├── tests/
│   ├── test_assistant.py
│   └── test_feedback.py
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
