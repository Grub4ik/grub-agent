# Grub Agent

A local voice assistant for Linux. Grub captures speech from a microphone, transcribes it with `faster-whisper`, sends the request through a small local agent using Ollama and explicitly registered tools, then prints, notifies, and optionally speaks the final reply with Silero TTS.

> A modular, local-first AI voice assistant for Linux, powered by local LLMs, speech recognition, and custom tools.

## Features

- Global X11 push-to-talk: hold F8 in any application to record; release it to transcribe.
- A short rising tone plays before capture and a distinct falling tone plays after capture.
- The start tone finishes before the microphone opens. Speak after hearing the tone.
- The complete model reply appears in the terminal and can be spoken by Silero.
- A desktop notification displays the final answer; intermediate tool calls are not notified or spoken.
- Terminal fallback remains available if global keyboard capture is unavailable.
- `faster-whisper` is loaded once and reused for the lifetime of the process.
- Ollama is called through its local HTTP API; conversation history is bounded by a configurable number of turns.
- Thinking output is disabled in Ollama requests so only the final answer is shown.
- Transcription time, LLM response time, and total processing latency are displayed.
- F8 callbacks only enqueue events. Recording coordination runs on the assistant thread, while STT and LLM work run in a worker thread.
- A failed recording, request, sound cue, or TTS operation does not terminate the assistant loop.
- Optional local Silero `v5_6_ru` speech synthesis, loaded once on startup and run on CPU.
- Selectable Silero voice (`aidar`, `eugene`, `baya`, `kseniya`, or `xenia`) and output volume/device.
- F8 can interrupt spoken playback and start a new push-to-talk recording.
- If PyTorch, Silero, or an output device is unavailable, text replies continue to work.
- Native Ollama tool calling with JSON Schema validation and an explicit registry/executor.
- Two read-only tools: system information and integer addition.
- `write` and `restricted` tools are denied by default.
- Tool execution is sequential, with limits of five tool rounds and eight tool calls per request.
- No shell-command or file-operation tool is registered.

## Requirements

- Linux Mint and Python 3.10 or newer (developed with Python 3.12).
- Ollama and the `gemma4:12b` model.
- `jsonschema` for validating model-generated tool arguments.
- `notify-send` from the `libnotify-bin` package for desktop notifications.
- A microphone accessible through PortAudio.
- An NVIDIA GPU with compatible CUDA/cuDNN runtime libraries for GPU-based `faster-whisper` inference.
- Internet access for the first Whisper model download. Transcription runs locally after the model is cached.
- Optional Silero TTS requires a compatible PyTorch installation. The first Silero model load downloads its model and Torch Hub code; subsequent runs use the local cache.

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

### Optional Silero TTS setup

Silero TTS uses PyTorch and OmegaConf, while the existing Whisper setup uses CTranslate2 and CUDA libraries. OmegaConf is included in `requirements.txt`; PyTorch is intentionally not pinned there because installing a generic PyTorch wheel can change an accelerator-specific setup. If you want speech output, install the PyTorch build appropriate for your operating system and CUDA/CPU choice using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Silero's documented standalone requirement is PyTorch 1.12 or newer. Keep the existing `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` dependencies used by faster-whisper.

The Silero model is loaded once when Grub starts. Its first load needs internet access to fetch the official Torch Hub repository and model weights; later loads use PyTorch's cache and can run offline. If PyTorch or Silero fails to load, Grub logs a warning and keeps text chat enabled.

> **Model licence:** the Silero repository currently lists the Russian V5 model under CC-NC-BY. Review the [official licence terms](https://github.com/snakers4/silero-models#licence) before redistributing it or using it commercially. This is separate from the Grub Agent source licence.

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

Use another voice or disable TTS:

```bash
python main.py --voice eugene
python main.py --no-tts
python main.py --voice kseniya --tts-volume 0.7 --playback-device 3
```

Run `main.py` from the project root using the Python environment where the requirements are installed. If you use the existing VoiceSTT environment, activate it first with `source ~/Projects/VoiceSTT/.venv/bin/activate`. Do not run `grub/assistant.py` directly; it is a package module, not an entry point. Calling `/usr/bin/python3` explicitly also bypasses an active virtual environment.

When Grub reports that it is ready, you can minimize the terminal and use other applications. Hold F8, wait for the start tone, speak, and release the key. The full answer appears in the terminal; Silero then speaks it. Press F8 while Grub is speaking to stop playback and begin a new recording. Press Ctrl+C in the terminal to exit. If F8 is unavailable, follow the Enter-mode prompts; terminal fallback does not support interrupting speech.

Tool calls are selected by the model through Ollama's native `/api/chat` `tools` field. Grub sends the tool schemas on each agent request, validates each argument object locally, checks permissions, and then invokes only the registered Python handler. The assistant's tool-call message and each `role: tool` result are kept in the conversation sent back to Ollama. Only the final answer is shown as a desktop notification or sent to TTS.

The registry contains exactly two read-only tools: `get_system_info` and `calculate_sum`. Tool definitions live with their handlers under `grub/tools/`; `main.py` registers them explicitly. No files are imported automatically from that folder.

The start tone is played before audio capture so it is not sent to Whisper. Start speaking after the tone. Sound output uses the system's default playback device independently of the selected microphone.

### Manual X11 check

1. Start Grub from a terminal in the X11 session and wait for the F8-ready message.
2. Minimize the terminal and open a browser or editor.
3. Hold F8, wait for the rising tone, say a short question, and release F8.
4. Confirm the falling tone plays, the full answer appears in the terminal, a desktop notification appears, and the reply is spoken.
5. During a spoken reply, press and keep holding F8 until the speech stops and the start tone plays; confirm that recording begins and release F8 after a short phrase.
6. Ask “Граб, какая у меня операционная система?” and confirm Grub calls `get_system_info` before answering.
7. Ask “Граб, сколько будет 125 плюс 347?” and confirm the final answer is based on `calculate_sum` returning `472`.
8. Ask “Граб, привет!” and confirm it answers without calling a tool.
9. Ask “Граб, удали все файлы на компьютере.” Confirm Grub explains that it cannot do this; there is no shell or file-operation tool available.

Tool selection depends on the model following the native Ollama tool interface; the executor still validates every call and will never execute a tool name that is not registered. These manual checks need a working Ollama instance and `gemma4:12b`.

The automated tests use fake LLM responses and stub audio. They do not simulate a physical key press, microphone speech, real Silero inference, or a live Ollama model.

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
| `GRUB_HISTORY_TURNS` | `10` | Number of completed conversation turns to retain, including tool messages |
| `GRUB_STT_MODEL` | `small` | `faster-whisper` model |
| `GRUB_STT_DEVICE` | `cuda` | `faster-whisper` device: `cuda` or `cpu` |
| `GRUB_STT_COMPUTE_TYPE` | `float16` | `faster-whisper` compute type |
| `GRUB_STT_LANGUAGE` | `ru` | Speech language; set to `auto` for automatic detection |
| `GRUB_AUDIO_DEVICE` | System default | Input device index from sounddevice |
| `GRUB_BEEP_VOLUME` | `0.12` | Feedback cue volume from `0.0` (muted) to `1.0` |
| `GRUB_FEEDBACK_DEVICE` | System default output | Optional sounddevice output device index |
| `GRUB_TTS_VOICE` | `aidar` | Silero voice: `aidar`, `eugene`, `baya`, `kseniya`, or `xenia` |
| `GRUB_TTS_VOLUME` | `1.0` | Speech playback volume from `0.0` to `1.0` |
| `GRUB_PLAYBACK_DEVICE` | System default output | Optional sounddevice output device index for speech |

Whisper and Gemma share GPU memory. If loading both causes a CUDA error or Ollama offloads part of the LLM to system memory, run speech recognition on the CPU:

```bash
GRUB_STT_DEVICE=cpu GRUB_STT_COMPUTE_TYPE=int8 python main.py
```

Automatic GPU memory management is not implemented. Moving STT to the CPU may increase transcription time.

Silero runs on CPU by default, so it does not claim GPU memory. Its synthesis work runs in the request worker, and speech playback uses its own output stream. The selected microphone and speech output device are independent.

### Try the Silero voices

Run the standalone audition tool (no Ollama, Whisper, or microphone is needed):

```bash
python tts_test.py --voice aidar
python tts_test.py --voice eugene --text "Проверка русского голоса и произношения." --volume 0.8
```

Omit `--text` to enter an arbitrary phrase interactively. The tool reports model load time, synthesis time, and generated audio duration. Use it to compare the five available voices.

## Tests and syntax check

```bash
python -m compileall -q main.py grub
python -m unittest discover -s tests -v
```

Unit tests cover tool registration, schema validation, permission denials, handler failures, native Ollama message formatting, normal replies, sequential tool calls, tool limits, and final-only notification/TTS behavior. They use fake LLM responses and do not test physical devices or real Silero inference.

## Project structure

```text
grub-agent/
├── grub/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── executor.py
│   │   ├── models.py
│   │   └── registry.py
│   ├── assistant.py
│   ├── audio.py
│   ├── feedback.py
│   ├── llm.py
│   ├── playback.py
│   ├── stt.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── calculate_sum.py
│   │   └── system_info.py
│   └── tts.py
├── tests/
│   ├── test_agent.py
│   ├── test_assistant.py
│   ├── test_feedback.py
│   ├── test_ollama_tools.py
│   └── test_tools.py
├── .gitignore
├── LICENSE
├── main.py
├── tts_test.py
├── README.md
└── requirements.txt
```

## Technologies

- Python — application language.
- sounddevice and NumPy — microphone capture, in-memory audio buffers, and output playback.
- faster-whisper / CTranslate2 — local speech recognition.
- Ollama HTTP API — local LLM chat.
- `jsonschema` — validation of model-generated tool arguments.
- Silero TTS / PyTorch (optional) — local Russian speech synthesis.
- pynput — F8 push-to-talk when supported by the desktop session.

## Roadmap

Wake word, VAD, streaming STT, GUI, additional write/restricted tools, MCP, vector memory, and autostart are not implemented yet.
