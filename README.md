# Grub Agent

A local voice assistant for Linux. Grub captures speech from a microphone, transcribes it with `faster-whisper`, sends the request through a small local agent using Ollama and explicitly registered tools, then prints and optionally speaks the final reply with Supertonic 3.

> A modular, local-first AI voice assistant for Linux, powered by local LLMs, speech recognition, and custom tools.

## Features

- Global X11 push-to-talk: hold F8 in any application to record; release it to transcribe.
- A short rising tone plays before capture and a distinct falling tone plays after capture.
- The start tone finishes before the microphone opens. Speak after hearing the tone.
- The complete model reply appears in the terminal and can be spoken by Supertonic 3.
- Terminal fallback remains available if global keyboard capture is unavailable.
- `faster-whisper` is loaded once and reused for the lifetime of the process.
- Ollama is called through its local HTTP API; conversation history is bounded by a configurable number of turns.
- Thinking output is disabled in Ollama requests so only the final answer is shown.
- Transcription time, LLM response time, and total processing latency are displayed.
- F8 callbacks only enqueue events. Recording coordination runs on the assistant thread, while STT and LLM work run in a worker thread.
- A failed recording, request, sound cue, or TTS operation does not terminate the assistant loop.
- Local Supertonic 3 is the speech engine. It runs with ONNX Runtime on CPU and does not compete with Gemma or Whisper for VRAM.
- Supertonic voice and language mode are configurable; `M1` and language-agnostic `na` are the defaults for Russian-English mixed text.
- TTS reports synthesis duration, audio duration, and real-time factor in the standalone audition tool.
- F8 can interrupt spoken playback and start a new push-to-talk recording.
- If Supertonic, its model, or an output device is unavailable, text replies continue to work.
- Native Ollama tool calling with JSON Schema validation and an explicit registry/executor.
- Eight core tools: system information, integer addition, SmartLED discovery, static RGB, rainbow, Music, Spectrum, and audio-mode stop. Four fact-memory tools are registered when ChromaDB initializes.
- `restricted` tools are denied; write permissions are denied except for the explicitly allowlisted SmartLED mode tools.
- Tool execution is sequential, with limits of five tool rounds and eight tool calls per request.
- No shell-command or file-operation tool is registered.
- Optional persistent user-fact memory uses ChromaDB and local Ollama embeddings; it stores short facts, not conversation transcripts.

## Requirements

- Linux Mint and Python 3.10 or newer (developed with Python 3.12).
- Ollama and the `gemma4:12b` model.
- `jsonschema` for validating model-generated tool arguments.
- `chromadb` for optional on-disk semantic memory.
- A microphone accessible through PortAudio.
- An NVIDIA GPU with compatible CUDA/cuDNN runtime libraries for GPU-based `faster-whisper` inference.
- Internet access for the first Whisper model download. Transcription runs locally after the model is cached.
- Supertonic 3 uses the official `supertonic` Python package and ONNX Runtime. Its model and voice styles are downloaded on first load (about 400 MB per upstream documentation) and cached locally.

F8 is handled globally by `pynput` without root access. Grub targets X11; Wayland may restrict global key capture. If the listener cannot start or exits, Grub prints the error and falls back to terminal Enter controls.

## Installation

```bash
cd ~/Projects/grub-agent
sudo apt install python3-venv python3-dev portaudio19-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### TTS setup

Grub uses the official [Supertonic Python SDK](https://github.com/supertone-oss-archive/supertonic-py), calling `TTS(model="supertonic-3", auto_download=True)`, `get_voice_style()`, and `synthesize()`. The upstream API returns a `(waveform, duration)` pair; Grub flattens the waveform to mono float samples and passes it to the existing interruptible `AudioPlayback` at the model's native 44.1 kHz sample rate. The `TTS` object and selected voice style are loaded once at startup and reused.

Supertonic uses ONNX Runtime's CPU execution provider. ONNX Runtime loads the model's ONNX graphs into inference sessions and executes their operators with CPU kernels; this TTS path does not initialize CUDA or require PyTorch. The first model load downloads about 400 MB of model and voice-style files into `~/.cache/supertonic3/`; later runs are local. Text replies remain available if the package, model, or audio output cannot load.

Choose a voice with `--voice M1` (built-in voices are `M1`–`M5` and `F1`–`F5`). Set `--tts-language ru` for Russian-specific handling, `en` for English, or `na` for the language-agnostic fallback; `na` is not a language detector and may pronounce mixed-language phrases differently from `ru`. Compare both modes with the standalone test below. Automatic language classification and perfect pronunciation of English words, IP addresses, and version numbers are not guaranteed.

**Maintenance note:** the upstream `supertone-oss-archive/supertonic-py` repository is archived (read-only as of September 9, 2026). The SDK is pinned to PyPI release `1.3.1` so API changes do not silently alter Grub. Do not expect upstream fixes or security updates from that repository.

Make sure Ollama is running and the model is available:

```bash
ollama pull gemma4:12b
ollama list
```

Run `ollama pull` only if the model has not already been downloaded. On its first run, `faster-whisper` downloads the `small` model and stores it in its local cache.

### Persistent fact memory

Grub can save stable facts the user shares about possessions, ongoing projects, and lasting preferences, then retrieve relevant facts by meaning in a later turn or session. It does not store full conversation transcripts. Sensitive information (such as passwords, secrets, health, or financial details) is not saved unless explicitly requested. You can ask what Grub remembers or ask it to forget a fact.

Memory uses ChromaDB's local `PersistentClient`, stored outside the repository at `~/.local/share/grub-agent/chroma` (or `$XDG_DATA_HOME/grub-agent/chroma`). The local Ollama embedding model defaults to `embeddinggemma`; install the Python package with `python -m pip install -r requirements.txt` and download the model once:

```bash
ollama pull embeddinggemma
```

Embedding requests use Ollama's local `/api/embed` endpoint, ask Ollama to run the embedding model on CPU (`num_gpu: 0`), and keep it loaded for ten minutes. This avoids taking VRAM from Gemma and faster-whisper. EmbeddingGemma requires Ollama 0.11.10 or newer. The first embedding request may take longer while Ollama loads the model. Once the model and Python package are present, memory works without internet access.

Configuration:

- `GRUB_MEMORY_PATH` — override the Chroma database directory.
- `GRUB_EMBEDDING_MODEL` — override the Ollama embedding model name.
- `GRUB_EMBEDDING_TIMEOUT` — embedding request timeout in seconds (default `20`).
- `GRUB_MEMORY_MAX_DISTANCE` — cosine-distance cutoff for retrieval (default `0.55`; tune against your own facts and queries).
- `GRUB_OLLAMA_URL` — Ollama base URL, shared with the chat client.

The first version uses model-guided fact selection: Grub is instructed to save durable facts, but an LLM can occasionally miss a fact or classify it incorrectly. Confirm important facts by asking what it remembers. Saved Chroma documents and metadata are stored locally as ordinary files; they are not encrypted by Grub.

Ask Grub to remember a durable fact naturally, then in a later session ask a related question. To review or remove facts, ask what it remembers or ask it to forget a specific item. Memory is optional: if Chroma cannot initialize, Grub continues without memory tools and prints a warning. If the embedding model is missing or Ollama is unavailable, the affected memory operation returns an error; ordinary chat remains available.

## Run

From the project root:

```bash
source .venv/bin/activate
python main.py
```

Select a Supertonic voice and language, or disable TTS:

```bash
python main.py --voice M1 --tts-language na
python main.py --voice F2 --tts-language ru
python main.py --no-tts
python main.py --voice M2 --tts-volume 0.7 --playback-device 3
```

Run `main.py` from the project root using the Python environment where the requirements are installed. If you use the existing VoiceSTT environment, activate it first with `source ~/Projects/VoiceSTT/.venv/bin/activate`. Do not run `grub/assistant.py` directly; it is a package module, not an entry point. Calling `/usr/bin/python3` explicitly also bypasses an active virtual environment.

When Grub reports that it is ready, you can minimize the terminal and use other applications. Hold F8, wait for the start tone, speak, and release the key. The full answer appears in the terminal; the selected TTS engine then speaks it. Press F8 while Grub is speaking to stop playback and begin a new recording. Press Ctrl+C in the terminal to exit. If F8 is unavailable, follow the Enter-mode prompts; terminal fallback does not support interrupting speech.

Tool calls are selected by the model through Ollama's native `/api/chat` `tools` field. Grub sends the tool schemas on each agent request, validates each argument object locally, checks permissions, and then invokes only the registered Python handler. The assistant's tool-call message and each `role: tool` result are kept in the conversation sent back to Ollama. Only the final answer is printed and optionally sent to TTS.

The registry explicitly registers `get_system_info`, `calculate_sum`, `discover_smartled`, `smartled_set_static`, `smartled_set_rainbow`, `smartled_start_music`, `smartled_start_spectrum`, and `smartled_stop_audio_mode`. Tool definitions live with their handlers under `grub/tools/`; no files are imported automatically from that folder.

`discover_smartled` checks only `192.168.0.1` through `192.168.0.254`, connects to TCP port `4210`, and sends only `HELP\n`. The model receives no arguments for this tool, so it cannot choose an address, port, command, or scan range. Each connection and response has a short timeout; with the defaults, the theoretical socket-wait budget for all 254 addresses is about 76 seconds, plus operating-system and loop overhead. A match stops the scan immediately. The discovered address is kept in the `SmartLEDTools` object for the lifetime of Grub and is not written to disk; a not-found scan clears the stored address.

`smartled_set_static` accepts integer `r`, `g`, and `b` channels in the range 0–255 (for example, red is `255, 0, 0`). The IP comes only from successful discovery; the model cannot provide an IP, port, or command. Grub sends `MODE STATIC r g b\n` through OpenBSD Netcat. `smartled_set_rainbow` accepts `interval` in milliseconds from 100 to 60000 and sends `MODE RAINBOW INTERVAL interval\n`.

`smartled_start_music` and `smartled_start_spectrum` start long-running system-audio loopback capture and send changing RGB datagrams to the discovered device. They follow LEDClient's analysis defaults and wire format: a TCP `MODE DYNAMIC` request must receive `OK`; then Grub samples the default speaker at 48 kHz in 1024-frame blocks and sends 5-byte UDP frames (`sequence` as big-endian uint16 followed by RGB) to port 4211 at up to 30 FPS. Music scales a configurable base RGB color by RMS brightness. Spectrum uses an 8-band FFT from 40 Hz to 10.24 kHz to select hue and audio level to select brightness; stereo channels are analyzed separately before their spectral power is combined, avoiding cancellation of out-of-phase audio. The two modes are mutually exclusive. Say “останови светомузыку” or “останови спектрум” to call `smartled_stop_audio_mode`; stopping ends capture and UDP updates, while the controller may retain the last received color until another mode command.

Music and Spectrum require the `SoundCard` dependency and a working PulseAudio server or PipeWire PulseAudio endpoint. Grub records the default speaker's loopback, not the microphone. The `write` executor allowlist names only the five SmartLED mode/start/stop tools; other write tools remain denied. Errors starting capture are returned as tool failures. No real device was contacted during implementation.

To test discovery manually, start Grub and ask “Граб, найди мою подсветку”; that starts the real local network scan. To request a write tool, ask “Теперь установи статический режим красного цвета” or “Включи режим радуги с интервалом 1000”. This sends a real device command; the automated tests use mocked subprocess calls and never connect to or control a device. Grub does not expose LEDClient's traffic-light command as a tool yet.

For loopback modes, first play audio on the default system output, then ask “включи светомузыку” or “включи спектрум”. Grub will capture that output and stream UDP frames until you say “останови светомузыку” or “останови спектрум”. These commands operate the actual controller. The new continuous audio modes have not been exercised against physical audio devices or a SmartLED controller.

The start tone is played before audio capture so it is not sent to Whisper. Start speaking after the tone. Sound output uses the system's default playback device independently of the selected microphone.

### Manual X11 check

1. Start Grub from a terminal in the X11 session and wait for the F8-ready message.
2. Minimize the terminal and open a browser or editor.
3. Hold F8, wait for the rising tone, say a short question, and release F8.
4. Confirm the falling tone plays, the full answer appears in the terminal, and the reply is spoken if TTS is enabled.
5. During a spoken reply, press and keep holding F8 until the speech stops and the start tone plays; confirm that recording begins and release F8 after a short phrase.
6. Ask “Граб, какая у меня операционная система?” and confirm Grub calls `get_system_info` before answering.
7. Ask “Граб, сколько будет 125 плюс 347?” and confirm the final answer is based on `calculate_sum` returning `472`.
8. Ask “Граб, привет!” and confirm it answers without calling a tool.
9. Ask “Граб, удали все файлы на компьютере.” Confirm Grub explains that it cannot do this; there is no shell or file-operation tool available.

Tool selection depends on the model following the native Ollama tool interface; the executor still validates every call and will never execute a tool name that is not registered. These manual checks need a working Ollama instance and `gemma4:12b`.

The automated tests use fake LLM responses and stub audio. They do not simulate a physical key press, microphone speech, real TTS model inference, or a live Ollama model.

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
| `GRUB_TTS_ENGINE` | `supertonic` | TTS engine |
| `GRUB_TTS_VOICE` | `M1` | Supertonic voice: `M1`–`M5` or `F1`–`F5` |
| `GRUB_TTS_LANGUAGE` | `na` | Text language mode: language-agnostic fallback, `ru`, or `en` |
| `GRUB_TTS_VOLUME` | `1.0` | Speech playback volume from `0.0` to `1.0` |
| `GRUB_PLAYBACK_DEVICE` | System default output | Optional sounddevice output device index for speech |

Whisper and Gemma share GPU memory. If loading both causes a CUDA error or Ollama offloads part of the LLM to system memory, run speech recognition on the CPU:

```bash
GRUB_STT_DEVICE=cpu GRUB_STT_COMPUTE_TYPE=int8 python main.py
```

Automatic GPU memory management is not implemented. Moving STT to the CPU may increase transcription time.

Supertonic synthesis runs on CPU and leaves GPU memory for Gemma and faster-whisper. TTS synthesis runs in the request worker, and playback remains on the existing interruptible audio stream. `--tts-device` accepts only `cpu` for compatibility with the current launch configuration.

### Test the Russian-English phrase

This standalone audition needs no Ollama, Whisper, or microphone. It downloads the model on first use, synthesizes and plays the phrase, and prints synthesis time, audio duration, and RTF (synthesis seconds divided by generated audio seconds):

```bash
python tts_test.py --voice M1 --compare-languages
```

The test synthesizes the same Russian-English technical phrase twice (`lang=na`, then `lang=ru`) and plays both clips so you can compare English words, the IP address, port, and `Python 3.12`. The program measures duration and RTF, but pronunciation quality must be judged by listening on your machine. To test one mode, use `python tts_test.py --voice M1 --language ru` (or `na` / `en`).

## Tests and syntax check

```bash
python -m compileall -q main.py grub
python -m unittest discover -s tests -v
```

Unit tests cover tool registration, schema validation, permission denials and per-tool write allowlisting, handler failures, native Ollama message formatting, normal replies, sequential tool calls, tool limits, SmartLED discovery with fake sockets, static and rainbow command handling with mocked subprocess, and Supertonic API behavior with stubs. The realtime Music/Spectrum capture and UDP worker are not covered by automated tests yet. Tests do not exercise physical devices, real TTS model inference/audio quality, or live network commands.

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
│   ├── memory.py
│   ├── playback.py
│   ├── stt.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── calculate_sum.py
│   │   ├── discover_smartled.py
│   │   ├── memory.py
│   │   ├── smartled.py
│   │   ├── smartled_audio.py
│   │   └── system_info.py
│   └── tts.py
├── tests/
│   ├── test_agent.py
│   ├── test_assistant.py
│   ├── test_discover_smartled.py
│   ├── test_smartled_static.py
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
- SoundCard — default speaker loopback capture for SmartLED Music and Spectrum modes.
- faster-whisper / CTranslate2 — local speech recognition.
- Ollama HTTP API — local LLM chat and local embeddings.
- ChromaDB (optional) — persistent vector storage for user facts.
- `jsonschema` — validation of model-generated tool arguments.
- Supertonic 3 / ONNX Runtime (optional) — local CPU speech synthesis.
- pynput — F8 push-to-talk when supported by the desktop session.

## Roadmap

Wake word, VAD, streaming STT, GUI, additional write/restricted tools, MCP, and autostart are not implemented yet.
