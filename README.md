# Grub Agent

Локальный голосовой ассистент для Linux. Grub записывает речь с микрофона, распознаёт её через `faster-whisper`, отправляет текст локальной модели Ollama и показывает ответ в терминале.

> A modular, local-first AI voice assistant for Linux, powered by local LLMs, speech recognition, and custom tools.

## Возможности v0.1

- Push-to-talk: удерживайте F8, чтобы говорить, и отпустите для распознавания.
- Резервный режим терминала: Enter начинает запись, следующий Enter останавливает её.
- Faster-whisper загружается один раз и повторно используется в течение запуска.
- Ollama вызывается через локальный HTTP API; история диалога ограничена числом ходов.
- Thinking выключен в запросах Ollama, чтобы в терминал попадал только ответ.
- Выводятся время распознавания, время ответа LLM и суммарная задержка обработки.
- Ошибка отдельной записи или запроса не завершает цикл ассистента.

## Требования

- Linux Mint, Python 3.10 или новее (разработка ориентирована на Python 3.12).
- Ollama и загруженная модель `gemma4:12b`.
- Микрофон, доступный через PortAudio.
- NVIDIA GPU и совместимые CUDA/cuDNN runtime-библиотеки для faster-whisper на GPU.
- Интернет при первой загрузке Whisper-модели; после кэширования распознавание работает локально.

F8 перехватывается через `pynput` без root-доступа. На X11 это обычно работает напрямую. Wayland может ограничивать глобальные клавиши; тогда Grub переключится на Enter-режим.

## Установка

```bash
cd ~/Projects/grub-agent
sudo apt install python3-venv python3-dev portaudio19-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Перед первым запуском убедитесь, что Ollama запущена и модель доступна:

```bash
ollama pull gemma4:12b
ollama list
```

Команда `ollama pull` нужна только если модель ещё не загружена. Whisper `small` скачивается faster-whisper при первом запуске и сохраняется в его локальном кэше.

## Запуск

Из корня проекта:

```bash
source .venv/bin/activate
python main.py
```

Когда появится сообщение о готовности, удерживайте F8, произнесите фразу и отпустите клавишу. Нажмите Ctrl+C для выхода. Если F8 недоступна, следуйте подсказкам Enter-режима.

Список устройств ввода можно посмотреть так:

```bash
python -c 'import sounddevice as sd; print(sd.query_devices())'
```

Чтобы выбрать устройство, задайте его номер:

```bash
GRUB_AUDIO_DEVICE=2 python main.py
```

## Настройки через переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `GRUB_MODEL` | `gemma4:12b` | Имя модели Ollama |
| `GRUB_OLLAMA_URL` | `http://localhost:11434` | Адрес локального API Ollama |
| `GRUB_OLLAMA_TIMEOUT` | `120` | Тайм-аут запроса в секундах |
| `GRUB_OLLAMA_KEEP_ALIVE` | `10m` | Как долго Ollama оставляет модель загруженной после запроса |
| `GRUB_HISTORY_TURNS` | `10` | Число сохранённых пар user/assistant |
| `GRUB_STT_MODEL` | `small` | Модель faster-whisper |
| `GRUB_STT_DEVICE` | `cuda` | Устройство faster-whisper: `cuda` или `cpu` |
| `GRUB_STT_COMPUTE_TYPE` | `float16` | Тип вычислений faster-whisper |
| `GRUB_STT_LANGUAGE` | `ru` | Язык речи; значение `auto` включает автоопределение |
| `GRUB_AUDIO_DEVICE` | системный ввод | Номер микрофона из списка sounddevice |

Whisper и Gemma используют одну видеопамять. Если совместная загрузка вызывает ошибку CUDA или Ollama переносит часть LLM в системную память, попробуйте запускать распознавание на CPU:

```bash
GRUB_STT_DEVICE=cpu GRUB_STT_COMPUTE_TYPE=int8 python main.py
```

Автоматического управления памятью GPU пока нет. Переход STT на CPU может увеличить время распознавания.

## Проверки

Unit-тесты не требуют микрофона, Ollama или загрузки моделей:

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py grub tests
```

## Структура проекта

```text
grub-agent/
├── grub/
│   ├── __init__.py
│   ├── assistant.py
│   ├── audio.py
│   ├── llm.py
│   └── stt.py
├── tests/
│   ├── test_assistant.py
│   └── test_llm.py
├── .gitignore
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

## Технологии

- Python — основной язык.
- sounddevice и NumPy — запись и хранение аудиобуферов в памяти.
- faster-whisper / CTranslate2 — локальное распознавание речи.
- Ollama HTTP API — локальный чат с LLM.
- pynput — F8 push-to-talk там, где это разрешает графическая сессия.

## План развития

TTS, wake word, VAD, streaming STT, GUI, инструменты и выполнение действий, MCP, векторная память и автозапуск пока не реализованы.
