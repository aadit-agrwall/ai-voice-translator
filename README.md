# 🌍 Real-Time Voice Translator

A fully **local, offline** voice translation tool built with open-source AI.  
Speak in one language → hear it translated into another in seconds.

```
Microphone → Whisper (STT) → Argos Translate → pyttsx3/Coqui (TTS) → Speaker
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Speech Recognition** | faster-whisper (Whisper v3) or Vosk |
| **Translation** | Argos Translate or MarianMT (HuggingFace) |
| **Text-to-Speech** | Coqui TTS (neural) or pyttsx3 (system) |
| **Voice Activity Detection** | webrtcvad – auto start/stop on speech |
| **34 Languages** | English, French, German, Spanish, Japanese, Chinese … |
| **GPU Acceleration** | CUDA auto-detected for Whisper + Coqui |
| **GUI** | Tkinter GUI with `--gui` flag |
| **Save Audio** | Export translated speech as WAV |
| **Streaming** | Continuous listening loop |

---

## 🗂 Project Structure

```
voice_translator/
├── main.py            # Entry point – CLI & GUI orchestration
├── speech_to_text.py  # STT backends (faster-whisper, Vosk)
├── translator.py      # Translation backends (Argos, MarianMT)
├── text_to_speech.py  # TTS backends (Coqui, pyttsx3)
├── audio_utils.py     # Microphone capture, VAD, playback
├── requirements.txt   # Python dependencies
└── README.md
```

---

## 🖥 System Requirements

| | Minimum | Recommended |
|---|---|---|
| **Python** | 3.9 | 3.11 |
| **RAM** | 4 GB | 8 GB |
| **Disk** | 2 GB | 5 GB |
| **GPU** | — | NVIDIA (CUDA 11.8+) |
| **OS** | Windows 10 / macOS 12 / Ubuntu 20.04 | Latest |

---

## 🚀 Installation

### 1. Clone / download the project

```bash
git clone https://github.com/yourname/voice_translator.git
cd voice_translator
```

### 2. Create a virtual environment (recommended)

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install core dependencies

```bash
pip install -r requirements.txt
```

### 4. OS-specific audio dependencies

**Ubuntu / Debian**
```bash
sudo apt-get install portaudio19-dev python3-dev espeak-ng
```

**macOS**
```bash
brew install portaudio espeak-ng
```

**Windows**
- Download and install [PortAudio](http://www.portaudio.com) (or install via conda: `conda install -c conda-forge portaudio`)
- espeak-ng for pyttsx3: https://github.com/espeak-ng/espeak-ng/releases

---

## 📦 Model Download Instructions

### faster-whisper (STT) – Auto download

Models are downloaded automatically on first run into `~/.cache/huggingface/`:

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `tiny` | 75 MB | ⚡⚡⚡⚡ | ★★☆☆ |
| `base` | 145 MB | ⚡⚡⚡ | ★★★☆ ← default |
| `small` | 461 MB | ⚡⚡ | ★★★★ |
| `medium` | 1.5 GB | ⚡ | ★★★★★ |
| `large-v3` | 3 GB | 🐢 | ★★★★★ |

```bash
# Pre-download without running (optional)
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
```

### Argos Translate (Translation) – Auto download

Language packages are downloaded automatically on first translation:
```bash
# Pre-install English ↔ French (example)
python -c "
import argostranslate.package as pkg
pkg.update_package_index()
available = pkg.get_available_packages()
en_fr = next(p for p in available if p.from_code=='en' and p.to_code=='fr')
pkg.install_from_path(en_fr.download())
print('Installed en→fr')
"
```

### Vosk (Alternative STT) – Manual download

1. Download a model from https://alphacephei.com/vosk/models
2. Extract to a local folder
3. Point to it with `--model-path`:

```bash
# Example: English model
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip -d models/

python main.py --stt vosk --model-path models/vosk-model-en-us-0.22
```

### Coqui TTS (Alternative TTS) – Auto download

```bash
pip install TTS

# List all available models
tts --list_models

# Pre-download XTTS v2 (best quality, ~2 GB)
tts --model_name "tts_models/multilingual/multi-dataset/xtts_v2" --text "test"
```

---

## ▶️ How to Run

### Basic (English → French)

```bash
python main.py --src en --tgt fr
```

### Interactive setup (pick languages in menus)

```bash
python main.py
```

### Launch GUI

```bash
python main.py --gui
```

### All CLI options

```bash
python main.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--src CODE` | (prompt) | Source language ISO code |
| `--tgt CODE` | (prompt) | Target language ISO code |
| `--stt` | `faster-whisper` | STT backend: `faster-whisper` or `vosk` |
| `--translation` | `argos` | Translation backend: `argos` or `marian` |
| `--tts` | `pyttsx3` | TTS backend: `pyttsx3` or `coqui` |
| `--whisper-model` | `base` | Whisper model size |
| `--model-path PATH` | — | Vosk model directory |
| `--coqui-model NAME` | `xtts_v2` | Coqui model name |
| `--device N` | (default) | Microphone device index |
| `--list-devices` | — | Print audio devices and exit |
| `--save-audio` | false | Save translated WAV files |
| `--gui` | false | Launch Tkinter GUI |
| `--verbose / -v` | false | Debug logging |

### More examples

```bash
# Japanese → English with GUI
python main.py --src ja --tgt en --gui

# High-quality TTS with Coqui
python main.py --src en --tgt de --tts coqui

# MarianMT translation + larger Whisper model
python main.py --src fr --tgt es --translation marian --whisper-model small

# Save all translated audio
python main.py --src en --tgt ru --save-audio

# List microphone devices, then select one
python main.py --list-devices
python main.py --device 2 --src en --tgt es
```

---

## 🔧 Troubleshooting

### `PortAudio not found` / `sounddevice error`
```bash
# Linux
sudo apt-get install portaudio19-dev
pip install --force-reinstall sounddevice pyaudio

# macOS
brew install portaudio
pip install --force-reinstall sounddevice
```

### `No module named 'webrtcvad'`
```bash
pip install webrtcvad-wheels   # pre-built, no C compiler needed
```
VAD is optional — the app still works without it (slightly less accurate silence detection).

### Argos Translate `ConnectionError` (no internet during first run)
Pre-download packages while you have internet; then use offline.

### `CUDA out of memory` with large Whisper model
Use a smaller model: `--whisper-model base` or `--whisper-model small`

### Coqui TTS very slow on CPU
XTTS v2 is GPU-optimised. On CPU use a lighter model:
```bash
python main.py --tts coqui --coqui-model "tts_models/en/ljspeech/tacotron2-DDC"
```
Or switch to pyttsx3 (instant, system voices):
```bash
python main.py --tts pyttsx3
```

### Windows: `pyttsx3` no audio output
Ensure the Windows Speech API (SAPI) is enabled and a voice is installed:  
Control Panel → Speech → Text-to-Speech → install a voice.

### `No speech detected` even when speaking
- Try a quieter environment
- Increase VAD aggressiveness in `audio_utils.py` (`VAD(aggressiveness=3)`)
- Check your microphone is the default input device: `python main.py --list-devices`

### Translation quality is poor
- Argos Translate uses smaller models; switch to MarianMT: `--translation marian`
- For non-English pairs, both backends bridge via English (e.g., fr→en→de); this is expected.

---

## 🌐 Supported Languages

| Language | Code | Language | Code |
|---|---|---|---|
| English | `en` | Japanese | `ja` |
| French | `fr` | Korean | `ko` |
| German | `de` | Chinese | `zh` |
| Spanish | `es` | Arabic | `ar` |
| Italian | `it` | Hindi | `hi` |
| Portuguese | `pt` | Russian | `ru` |
| Dutch | `nl` | Turkish | `tr` |
| Polish | `pl` | Ukrainian | `uk` |
| … and 18 more | | | |

Run `python main.py` and choose from the interactive menu to see all 34 languages.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       main.py                               │
│  VoiceTranslator – orchestrates pipeline + CLI/GUI          │
└──────────────┬──────────────┬───────────────┬───────────────┘
               │              │               │
    ┌──────────▼───┐  ┌───────▼───────┐  ┌───▼──────────────┐
    │ audio_utils  │  │speech_to_text │  │  translator.py   │
    │              │  │               │  │                  │
    │ AudioRecorder│  │FasterWhisper  │  │ ArgosTranslator  │
    │ VAD (webrtc) │  │VoskSTT        │  │ MarianMTTranslat │
    │ play_audio() │  │               │  │                  │
    └──────────────┘  └───────────────┘  └──────────────────┘
                                                  │
                                         ┌────────▼────────┐
                                         │text_to_speech.py│
                                         │                 │
                                         │ CoquiTTS        │
                                         │ Pyttsx3TTS      │
                                         └─────────────────┘
```

---

## 📄 License

MIT – free for personal and commercial use.

---

## 🙏 Credits

- [OpenAI Whisper](https://github.com/openai/whisper) + [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [Argos Translate](https://github.com/argosopentech/argos-translate)
- [Helsinki-NLP MarianMT](https://huggingface.co/Helsinki-NLP)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Vosk](https://alphacephei.com/vosk/)
- [webrtcvad](https://github.com/wiseman/py-webrtcvad)
