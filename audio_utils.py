"""
audio_utils.py
==============
Handles microphone recording, Voice Activity Detection (VAD),
and audio playback utilities.

Uses sounddevice for cross-platform audio I/O and webrtcvad for
lightweight, fast Voice Activity Detection.
"""

import logging
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 16_000   # Hz – required by Whisper & Vosk
DEFAULT_CHANNELS = 1            # Mono
DEFAULT_DTYPE = "int16"         # 16-bit PCM
FRAME_DURATION_MS = 30          # VAD frame size in ms (10, 20, or 30)
CHUNK_SIZE = int(DEFAULT_SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # samples per frame
SILENCE_TIMEOUT = 1.5           # seconds of silence before ending utterance
MAX_UTTERANCE_SECONDS = 30      # hard cap per recording


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def list_audio_devices() -> None:
    """Print all available audio input/output devices."""
    print("\n📢  Available Audio Devices")
    print("=" * 50)
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        kind = []
        if dev["max_input_channels"] > 0:
            kind.append("IN")
        if dev["max_output_channels"] > 0:
            kind.append("OUT")
        print(f"  [{idx:2d}] {dev['name']}  ({'/'.join(kind)})")
    print(f"\n  Default input : {sd.query_devices(kind='input')['name']}")
    print(f"  Default output: {sd.query_devices(kind='output')['name']}\n")


def get_default_input_device() -> Optional[int]:
    """Return the default input device index, or None if not found."""
    try:
        return sd.query_devices(kind="input")["index"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Voice Activity Detection
# ---------------------------------------------------------------------------

class VAD:
    """
    Thin wrapper around webrtcvad.Vad with helper utilities.

    Aggressiveness levels:
        0 – least aggressive (keeps more audio as speech)
        3 – most aggressive (filters out more non-speech)
    """

    def __init__(self, aggressiveness: int = 2):
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(aggressiveness)
            self._available = True
            logger.info("VAD initialised (aggressiveness=%d)", aggressiveness)
        except ImportError:
            logger.warning(
                "webrtcvad not installed – VAD disabled. "
                "Install with: pip install webrtcvad-wheels"
            )
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def is_speech(self, frame: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bool:
        """Return True if the audio frame contains speech."""
        if not self._available:
            return True  # assume speech when VAD unavailable
        try:
            return self._vad.is_speech(frame, sample_rate)
        except Exception:
            return True


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class AudioRecorder:
    """
    Records audio from the microphone using VAD to detect speech boundaries.

    Usage:
        recorder = AudioRecorder()
        audio_bytes = recorder.record_utterance()   # blocks until utterance ends
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        device: Optional[int] = None,
        vad_aggressiveness: int = 2,
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.vad = VAD(aggressiveness=vad_aggressiveness)
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_utterance(
        self,
        on_start: Optional[Callable] = None,
        on_end: Optional[Callable] = None,
    ) -> bytes:
        """
        Block until a complete voice utterance is captured.

        Returns raw PCM bytes (int16, mono, DEFAULT_SAMPLE_RATE).
        """
        audio_queue: queue.Queue = queue.Queue()
        self._stop_event.clear()

        def callback(indata, frames, time_info, status):
            if status:
                logger.debug("sounddevice status: %s", status)
            audio_queue.put(indata.copy())

        frames_buffer: list[np.ndarray] = []
        speech_frames: list[np.ndarray] = []
        in_speech = False
        silence_start: Optional[float] = None
        recording_start: Optional[float] = None

        logger.info("Listening for speech…")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=DEFAULT_CHANNELS,
            dtype=DEFAULT_DTYPE,
            blocksize=CHUNK_SIZE,
            device=self.device,
            callback=callback,
        ):
            while not self._stop_event.is_set():
                try:
                    frame = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                frame_bytes = frame.tobytes()
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)

                if is_speech:
                    if not in_speech:
                        # Speech onset
                        in_speech = True
                        recording_start = time.time()
                        silence_start = None
                        if on_start:
                            on_start()
                        logger.debug("Speech started")
                        # Include a small pre-buffer for natural start
                        speech_frames = frames_buffer[-5:] + [frame]
                    else:
                        speech_frames.append(frame)
                        silence_start = None
                else:
                    frames_buffer.append(frame)
                    if len(frames_buffer) > 10:
                        frames_buffer.pop(0)

                    if in_speech:
                        speech_frames.append(frame)
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > SILENCE_TIMEOUT:
                            logger.debug("Speech ended (silence timeout)")
                            break

                # Hard cap
                if recording_start and (time.time() - recording_start > MAX_UTTERANCE_SECONDS):
                    logger.debug("Speech ended (max duration)")
                    break

        if on_end:
            on_end()

        if not speech_frames:
            return b""

        audio_np = np.concatenate(speech_frames, axis=0)
        return audio_np.tobytes()

    def stop(self) -> None:
        """Signal the recorder to stop (used for hotkey / GUI control)."""
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def play_audio_file(path: str | Path) -> None:
    """
    Play an audio file (WAV / MP3 / OGG) through the default output device.
    Blocks until playback is complete.
    """
    path = Path(path)
    if not path.exists():
        logger.error("Audio file not found: %s", path)
        return

    try:
        data, sr = sf.read(str(path), dtype="float32")
        sd.play(data, samplerate=sr)
        sd.wait()
        logger.debug("Playback complete: %s", path)
    except Exception as exc:
        logger.error("Playback error: %s", exc)


def play_audio_bytes(audio_bytes: bytes, sample_rate: int = 22_050) -> None:
    """
    Play raw float32 PCM bytes through the default output device.
    """
    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        sd.play(audio_np, samplerate=sample_rate)
        sd.wait()
    except Exception as exc:
        logger.error("Playback error: %s", exc)


def save_audio_bytes_as_wav(
    audio_bytes: bytes,
    path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sampwidth: int = 2,
) -> Path:
    """
    Save raw PCM int16 bytes to a WAV file.

    Returns the saved path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    logger.info("Saved WAV: %s", path)
    return path
