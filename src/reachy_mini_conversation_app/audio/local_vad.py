import gi
import numpy as np
from numpy.typing import NDArray


gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


class WebRTCVAD:
    """Detect speech in captured microphone audio for physical listening reactions."""

    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 160
    SILENCE_SAMPLES = 6400  # 400 ms

    def __init__(self) -> None:
        """Start GStreamer's WebRTC detector without additional audio processing."""
        Gst.init(None)
        self._pipeline = Gst.parse_launch(
            "appsrc name=source is-live=true format=time "
            "caps=audio/x-raw,format=S16LE,rate=16000,channels=1,layout=interleaved "
            "! webrtcdsp name=detector voice-detection=true echo-cancel=false "
            "gain-control=false noise-suppression=false high-pass-filter=false "
            "! appsink name=sink sync=false async=false max-buffers=1"
        )
        self._source = self._pipeline.get_by_name("source")
        self._sink = self._pipeline.get_by_name("sink")
        self._bus = self._pipeline.get_bus()
        if not self._pipeline.get_by_name("detector").get_property("voice-detection"):
            self.close()
            raise RuntimeError("Installed webrtcdsp does not support voice detection")
        self.reset()

    def reset(self) -> None:
        """Clear the detector and buffered audio between microphone streams."""
        self._pipeline.set_state(Gst.State.NULL)
        self._pending = np.empty(0, dtype=np.int16)
        self._voice = False
        self._speaking = False
        self._silence_samples = 0
        self._timestamp = 0
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self.close()
            raise RuntimeError("Failed to start WebRTC VAD pipeline")

    def close(self) -> None:
        """Release GStreamer resources when local detection stops."""
        self._pipeline.set_state(Gst.State.NULL)

    def process(self, audio: NDArray[np.int16], sample_rate: int, *, reset: bool = False) -> bool:
        """Return speech activity, holding through 400 ms of silence."""
        if sample_rate != self.SAMPLE_RATE:
            raise ValueError("Local VAD requires 16 kHz microphone audio")
        if reset:
            self.reset()
        self._pending = np.concatenate((self._pending, audio))
        offset = 0
        while self._pending.size - offset >= self.CHUNK_SAMPLES:
            chunk = self._pending[offset : offset + self.CHUNK_SAMPLES].astype("<i2", copy=False).tobytes()
            buffer = Gst.Buffer.new_allocate(None, len(chunk), None)
            buffer.fill(0, chunk)
            buffer.pts = self._timestamp
            buffer.duration = self.CHUNK_SAMPLES * Gst.SECOND // self.SAMPLE_RATE
            self._timestamp += buffer.duration
            if self._source.emit("push-buffer", buffer) != Gst.FlowReturn.OK:
                raise RuntimeError("WebRTC VAD rejected microphone audio")
            # Wait off the event loop so the speech event belongs to this audio chunk.
            sample = self._sink.emit("try-pull-sample", Gst.SECOND)
            while message := self._bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.ELEMENT):
                if message.type == Gst.MessageType.ERROR:
                    error, detail = message.parse_error()
                    raise RuntimeError("WebRTC VAD pipeline failed: %s; %s" % (error, detail))
                structure = message.get_structure()
                if structure is not None and structure.get_name() == "voice-activity":
                    self._voice = bool(structure.get_value("stream-has-voice"))
            if sample is None:
                raise TimeoutError("WebRTC VAD did not process microphone audio")
            if self._voice:
                self._speaking = True
                self._silence_samples = 0
            elif self._speaking:
                self._silence_samples += self.CHUNK_SAMPLES
                if self._silence_samples >= self.SILENCE_SAMPLES:
                    self._speaking = False
                    self._silence_samples = 0
            offset += self.CHUNK_SAMPLES
        self._pending = self._pending[offset:].copy()
        return self._speaking
