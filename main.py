"""
main.py
=======
Real-Time Voice Translator – Entry Point
=========================================

CLI and optional Tkinter GUI for the voice translation pipeline.

Pipeline:
    Microphone → STT → Translation → TTS → Speaker

Usage:
    python main.py                          # Interactive CLI
    python main.py --gui                    # Tkinter GUI
    python main.py --src en --tgt fr        # Skip language prompt
    python main.py --stt vosk --tts pyttsx3 # Choose backends
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# ── Local modules ──────────────────────────────────────────────────────────
from audio_utils import AudioRecorder, play_audio_file, list_audio_devices
from speech_to_text import create_stt_backend, STTBackend
from translator import create_translator, TranslationBackend, LANGUAGES, get_language_name
from text_to_speech import create_tts_backend, TTSBackend

# ── Logging ────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("voice_translator.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# ===========================================================================
# Core pipeline
# ===========================================================================

class VoiceTranslator:
    """
    Orchestrates the full voice translation pipeline.

    Attributes
    ----------
    stt      : STTBackend        – converts audio bytes to text
    tl       : TranslationBackend – translates text
    tts      : TTSBackend        – converts text to audio
    recorder : AudioRecorder     – captures microphone input
    """

    def __init__(
        self,
        stt: STTBackend,
        tl: TranslationBackend,
        tts: TTSBackend,
        src_lang: str = "en",
        tgt_lang: str = "fr",
        save_audio: bool = False,
        audio_device: Optional[int] = None,
    ):
        self.stt = stt
        self.tl = tl
        self.tts = tts
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.save_audio = save_audio
        self.recorder = AudioRecorder(device=audio_device)

        self._running = False
        self._stop_hotkey_thread: Optional[threading.Thread] = None

        # Optional callback for GUI updates
        self.on_stt_result: Optional[callable] = None
        self.on_translation: Optional[callable] = None

    # ------------------------------------------------------------------
    # Pipeline step
    # ------------------------------------------------------------------

    def process_utterance(self, audio_bytes: bytes) -> tuple[str, str]:
        """
        Run one complete STT → translate → TTS → play cycle.

        Returns (original_text, translated_text).
        """
        if not audio_bytes:
            return "", ""

        # 1. Speech → Text
        print("\n⏳  Transcribing …", end=" ", flush=True)
        original_text = self.stt.transcribe(audio_bytes, language=self.src_lang)
        if not original_text.strip():
            print("(no speech detected)")
            return "", ""
        print("✓")

        print(f"🎤  [{get_language_name(self.src_lang)}] {original_text}")
        if self.on_stt_result:
            self.on_stt_result(original_text)

        # 2. Translate
        print("⏳  Translating …", end=" ", flush=True)
        translated_text = self.tl.translate(original_text, self.src_lang, self.tgt_lang)
        print("✓")
        print(f"🌐  [{get_language_name(self.tgt_lang)}] {translated_text}")
        if self.on_translation:
            self.on_translation(translated_text)

        # 3. Text → Speech
        print("⏳  Synthesising speech …", end=" ", flush=True)
        output_dir = Path("output_audio") if self.save_audio else Path(
            os.environ.get("TMPDIR", "/tmp")
        )
        output_path = output_dir / f"translated_{int(time.time())}.wav"
        wav_path = self.tts.synthesize(translated_text, language=self.tgt_lang, output_path=output_path)
        print("✓")

        # 4. Play audio
        print("🔊  Playing …", end=" ", flush=True)
        play_audio_file(wav_path)
        print("✓")

        # Optionally remove temp file
        if not self.save_audio and wav_path.exists():
            try:
                wav_path.unlink()
            except Exception:
                pass

        return original_text, translated_text

    # ------------------------------------------------------------------
    # Continuous listening loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the continuous listen → translate loop (CLI)."""
        self._running = True

        print("\n" + "=" * 60)
        print("  🎙️  Real-Time Voice Translator")
        print("=" * 60)
        print(f"  Source  : {get_language_name(self.src_lang)} ({self.src_lang})")
        print(f"  Target  : {get_language_name(self.tgt_lang)} ({self.tgt_lang})")
        print(f"  STT     : {self.stt.__class__.__name__}")
        print(f"  TTS     : {self.tts.__class__.__name__}")
        print("  Press Ctrl+C to stop")
        print("=" * 60)

        try:
            while self._running:
                print("\n🎤  Listening … (speak now)")
                audio_bytes = self.recorder.record_utterance(
                    on_start=lambda: print("\n🔴  Recording …"),
                    on_end=lambda: print("⏹  Processing …"),
                )
                if audio_bytes:
                    self.process_utterance(audio_bytes)
                else:
                    logger.debug("Empty audio; skipping.")
        except KeyboardInterrupt:
            print("\n\n👋  Stopped by user. Goodbye!")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
        self.recorder.stop()


# ===========================================================================
# CLI helpers
# ===========================================================================

def pick_language(prompt: str, default: str = "en") -> str:
    """Interactive language selection menu."""
    items = sorted(LANGUAGES.items())
    print(f"\n{prompt}")
    print("-" * 40)
    for idx, (name, code) in enumerate(items, 1):
        print(f"  {idx:2d}. {name:<30} [{code}]")
    print("-" * 40)
    while True:
        raw = input(f"  Enter number or code [default: {default}]: ").strip()
        if not raw:
            return default
        # Code entered directly
        if raw in LANGUAGES.values():
            return raw
        # Number entered
        try:
            n = int(raw)
            if 1 <= n <= len(items):
                return items[n - 1][1]
        except ValueError:
            pass
        print("  ❌ Invalid selection, please try again.")


def print_banner() -> None:
    banner = r"""
  ╔══════════════════════════════════════════╗
  ║   🌍  Real-Time Voice Translator  🎙️     ║
  ║        Powered by Open-Source AI          ║
  ╚══════════════════════════════════════════╝
"""
    print(banner)


# ===========================================================================
# Tkinter GUI (bonus)
# ===========================================================================

def launch_gui(translator: VoiceTranslator) -> None:
    """Launch a simple Tkinter GUI for the translator."""
    try:
        import tkinter as tk
        from tkinter import ttk, scrolledtext
    except ImportError:
        print("Tkinter is not available. Falling back to CLI mode.")
        translator.run()
        return

    root = tk.Tk()
    root.title("🌍 Real-Time Voice Translator")
    root.geometry("700x500")
    root.resizable(True, True)

    # ── Style ──────────────────────────────────────────────────────────
    style = ttk.Style()
    style.theme_use("clam")

    BG = "#1e1e2e"
    FG = "#cdd6f4"
    ACCENT = "#89b4fa"
    GREEN = "#a6e3a1"
    YELLOW = "#f9e2af"
    root.configure(bg=BG)

    # ── Header ─────────────────────────────────────────────────────────
    header = tk.Label(
        root, text="🌍 Real-Time Voice Translator",
        font=("Helvetica", 18, "bold"),
        bg=BG, fg=ACCENT,
    )
    header.pack(pady=(15, 5))

    lang_frame = tk.Frame(root, bg=BG)
    lang_frame.pack(pady=5)

    tk.Label(lang_frame, text="Source:", bg=BG, fg=FG).grid(row=0, column=0, padx=5)
    src_var = tk.StringVar(value=translator.src_lang)
    src_cb = ttk.Combobox(
        lang_frame, textvariable=src_var,
        values=[f"{name} ({code})" for name, code in sorted(LANGUAGES.items())],
        width=20, state="readonly",
    )
    src_cb.grid(row=0, column=1, padx=5)

    tk.Label(lang_frame, text="→", bg=BG, fg=FG, font=("Helvetica", 14)).grid(row=0, column=2, padx=5)

    tk.Label(lang_frame, text="Target:", bg=BG, fg=FG).grid(row=0, column=3, padx=5)
    tgt_var = tk.StringVar(value=translator.tgt_lang)
    tgt_cb = ttk.Combobox(
        lang_frame, textvariable=tgt_var,
        values=[f"{name} ({code})" for name, code in sorted(LANGUAGES.items())],
        width=20, state="readonly",
    )
    tgt_cb.grid(row=0, column=4, padx=5)

    # ── Text display ───────────────────────────────────────────────────
    text_frame = tk.Frame(root, bg=BG)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    tk.Label(text_frame, text="🎤 Original", bg=BG, fg=YELLOW, font=("Helvetica", 11, "bold")).pack(anchor="w")
    orig_box = scrolledtext.ScrolledText(
        text_frame, height=5, font=("Helvetica", 12),
        bg="#313244", fg=FG, insertbackground=FG, wrap=tk.WORD,
    )
    orig_box.pack(fill=tk.BOTH, expand=True, pady=(2, 10))

    tk.Label(text_frame, text="🌐 Translated", bg=BG, fg=GREEN, font=("Helvetica", 11, "bold")).pack(anchor="w")
    trans_box = scrolledtext.ScrolledText(
        text_frame, height=5, font=("Helvetica", 12),
        bg="#313244", fg=FG, insertbackground=FG, wrap=tk.WORD,
    )
    trans_box.pack(fill=tk.BOTH, expand=True, pady=(2, 5))

    # ── Status + buttons ───────────────────────────────────────────────
    status_var = tk.StringVar(value="Ready. Press Start to listen.")
    status_label = tk.Label(root, textvariable=status_var, bg=BG, fg=ACCENT, font=("Helvetica", 10))
    status_label.pack(pady=5)

    btn_frame = tk.Frame(root, bg=BG)
    btn_frame.pack(pady=10)

    is_listening = threading.Event()

    def update_orig(text: str) -> None:
        orig_box.delete("1.0", tk.END)
        orig_box.insert(tk.END, text)

    def update_trans(text: str) -> None:
        trans_box.delete("1.0", tk.END)
        trans_box.insert(tk.END, text)

    translator.on_stt_result = lambda t: root.after(0, update_orig, t)
    translator.on_translation = lambda t: root.after(0, update_trans, t)

    def start_listening() -> None:
        if is_listening.is_set():
            return

        # Update language from dropdowns
        src_txt = src_var.get()
        tgt_txt = tgt_var.get()
        # Extract code from "Name (code)" format
        translator.src_lang = src_txt.split("(")[-1].rstrip(")")
        translator.tgt_lang = tgt_txt.split("(")[-1].rstrip(")")

        is_listening.set()
        start_btn.config(state=tk.DISABLED)
        stop_btn.config(state=tk.NORMAL)
        status_var.set("🔴 Listening …")
        thread = threading.Thread(target=listen_loop, daemon=True)
        thread.start()

    def stop_listening() -> None:
        is_listening.clear()
        translator.stop()
        start_btn.config(state=tk.NORMAL)
        stop_btn.config(state=tk.DISABLED)
        status_var.set("Stopped. Press Start to listen again.")

    def listen_loop() -> None:
        while is_listening.is_set():
            root.after(0, status_var.set, "🎤 Speak now …")
            audio_bytes = translator.recorder.record_utterance(
                on_start=lambda: root.after(0, status_var.set, "🔴 Recording …"),
                on_end=lambda: root.after(0, status_var.set, "⏳ Processing …"),
            )
            if audio_bytes and is_listening.is_set():
                translator.process_utterance(audio_bytes)
                root.after(0, status_var.set, "✅ Done. Listening again …")

    btn_style = {
        "font": ("Helvetica", 12, "bold"),
        "relief": "flat",
        "cursor": "hand2",
        "padx": 20, "pady": 8,
    }

    start_btn = tk.Button(
        btn_frame, text="▶  Start", bg=GREEN, fg="#1e1e2e",
        command=start_listening, **btn_style,
    )
    start_btn.grid(row=0, column=0, padx=10)

    stop_btn = tk.Button(
        btn_frame, text="⏹  Stop", bg="#f38ba8", fg="#1e1e2e",
        command=stop_listening, state=tk.DISABLED, **btn_style,
    )
    stop_btn.grid(row=0, column=1, padx=10)

    # Bind Enter key to start/stop toggle
    root.bind("<Return>", lambda _: start_listening() if not is_listening.is_set() else stop_listening())

    root.mainloop()


# ===========================================================================
# Argument parsing & main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-Time Voice Translator – fully local, open-source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Interactive setup
  python main.py --src en --tgt fr            # English → French
  python main.py --src ja --tgt en --gui      # Japanese → English (GUI)
  python main.py --stt vosk --model-path ./models/vosk-en
  python main.py --translation marian --tts coqui
        """,
    )

    # Language
    p.add_argument("--src", default=None, help="Source language ISO code (e.g. en, fr, de)")
    p.add_argument("--tgt", default=None, help="Target language ISO code (e.g. fr, es, ja)")

    # Backends
    p.add_argument("--stt", default="faster-whisper", choices=["faster-whisper", "vosk"],
                   help="Speech recognition backend (default: faster-whisper)")
    p.add_argument("--translation", default="argos", choices=["argos", "marian"],
                   help="Translation backend (default: argos)")
    p.add_argument("--tts", default="pyttsx3", choices=["coqui", "pyttsx3"],
                   help="Text-to-speech backend (default: pyttsx3)")

    # STT options
    p.add_argument("--whisper-model", default="base",
                   choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                   help="Whisper model size (default: base)")
    p.add_argument("--model-path", default=None,
                   help="Path to Vosk model directory")

    # TTS options
    p.add_argument("--coqui-model", default="tts_models/multilingual/multi-dataset/xtts_v2",
                   help="Coqui TTS model name")

    # Audio
    p.add_argument("--device", type=int, default=None, help="Microphone device index")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")

    # Misc
    p.add_argument("--save-audio", action="store_true", help="Save translated audio files")
    p.add_argument("--gui", action="store_true", help="Launch Tkinter GUI")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    return p.parse_args()


def build_translator(args: argparse.Namespace, src_lang: str, tgt_lang: str) -> VoiceTranslator:
    """Instantiate all backends and return a VoiceTranslator."""

    # STT
    if args.stt == "faster-whisper":
        stt = create_stt_backend("faster-whisper", model_size=args.whisper_model)
    else:
        if not args.model_path:
            print("\n❌  Vosk requires --model-path pointing to your downloaded model directory.")
            print("    Download models from: https://alphacephei.com/vosk/models")
            sys.exit(1)
        stt = create_stt_backend("vosk", model_path=args.model_path)

    # Translator
    tl = create_translator(args.translation)

    # TTS
    if args.tts == "coqui":
        tts = create_tts_backend("coqui", model_name=args.coqui_model)
    else:
        tts = create_tts_backend("pyttsx3")

    # Warm up all backends
    print("\n⏳  Initialising models (first run may download weights) …")
    try:
        stt.warmup()
        tl.warmup()
        tts.warmup()
    except Exception as exc:
        print(f"\n❌  Initialisation failed: {exc}")
        print("    Check the README for installation instructions.")
        sys.exit(1)
    print("✅  All backends ready.\n")

    return VoiceTranslator(
        stt=stt,
        tl=tl,
        tts=tts,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        save_audio=args.save_audio,
        audio_device=args.device,
    )


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print_banner()

    # List devices and exit
    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    # Language selection
    src_lang = args.src or pick_language("Select SOURCE language (what you will speak):", "en")
    tgt_lang = args.tgt or pick_language("Select TARGET language (translation output):", "fr")

    if src_lang not in LANGUAGES.values():
        print(f"❌  Unknown source language code: '{src_lang}'")
        sys.exit(1)
    if tgt_lang not in LANGUAGES.values():
        print(f"❌  Unknown target language code: '{tgt_lang}'")
        sys.exit(1)

    print(f"\n📌  {get_language_name(src_lang)} → {get_language_name(tgt_lang)}")

    vt = build_translator(args, src_lang, tgt_lang)

    if args.gui:
        launch_gui(vt)
    else:
        vt.run()


if __name__ == "__main__":
    main()
