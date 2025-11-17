"""Deepgram streaming ASR provider."""

from __future__ import annotations
import io
import wave
import asyncio
import logging
from typing import Any, Optional

from .base_streaming import StreamingASRProvider


logger = logging.getLogger(__name__)


class DeepgramStreamingASR(StreamingASRProvider):
    """Deepgram streaming ASR implementation using WebSocket."""

    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
        language: str = "en",
    ):
        """Initialize Deepgram streaming ASR.

        Args:
            api_key: Deepgram API key
            model: Model to use (nova-2 recommended for best quality/speed)
            language: Language code (e.g., 'en', 'fr')

        """
        self.api_key = api_key
        self.model = model
        self.language = language

        # Streaming state
        self.connection = None
        self.partial_transcript = ""
        self.final_transcript = ""
        self.transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self.is_final_received = False

        # Import Deepgram SDK
        try:
            from deepgram import (
                LiveOptions,
                DeepgramClient,
                DeepgramClientOptions,
                LiveTranscriptionEvents,
            )

            self.DeepgramClient = DeepgramClient
            self.DeepgramClientOptions = DeepgramClientOptions
            self.LiveOptions = LiveOptions
            self.LiveTranscriptionEvents = LiveTranscriptionEvents
        except ImportError:
            raise ImportError(
                "Deepgram SDK not installed. Install with: pip install deepgram-sdk"
            )

        logger.info(f"Deepgram streaming ASR initialized (model={model}, language={language})")

    async def start_stream(self) -> None:
        """Initialize streaming session with Deepgram."""
        try:
            # Create Deepgram client
            config = self.DeepgramClientOptions(
                options={"keepalive": "true"},
            )
            client = self.DeepgramClient(self.api_key, config)

            # Get live transcription connection
            self.connection = client.listen.asyncwebsocket.v("1")
            assert self.connection is not None  # Type narrowing for mypy

            # Set up event handlers
            self.connection.on(
                self.LiveTranscriptionEvents.Transcript,
                self._on_message,
            )

            self.connection.on(
                self.LiveTranscriptionEvents.Error,
                self._on_error,
            )

            # Configure live transcription options
            options = self.LiveOptions(
                model=self.model,
                language=self.language,
                encoding="linear16",
                sample_rate=16000,
                channels=1,
                interim_results=True,  # Get partial transcripts
                punctuate=True,
                smart_format=True,
            )

            # Start the connection
            if await self.connection.start(options):
                logger.info("Deepgram WebSocket connection established")
                # Reset state
                self.partial_transcript = ""
                self.final_transcript = ""
                self.is_final_received = False
                self.transcript_queue = asyncio.Queue()
            else:
                raise RuntimeError("Failed to start Deepgram connection")

        except Exception as e:
            logger.exception(f"Failed to start Deepgram stream: {e}")
            raise

    async def send_audio_chunk(self, audio_chunk: bytes) -> None:
        """Send audio chunk to Deepgram.

        Args:
            audio_chunk: Audio data (WAV format will be converted to raw PCM)

        """
        if not self.connection:
            logger.warning("Deepgram connection not established, skipping audio chunk")
            return

        try:
            # Convert WAV to raw PCM if needed
            pcm_data = self._wav_to_pcm(audio_chunk)

            # Send to Deepgram
            await self.connection.send(pcm_data)

        except Exception as e:
            logger.warning(f"Failed to send audio chunk to Deepgram: {e}")

    async def get_partial_transcript(self) -> Optional[str]:
        """Get current partial transcript.

        Returns:
            Current partial transcript, or None if nothing available yet

        """
        return self.partial_transcript if self.partial_transcript else None

    async def end_stream(self) -> str:
        """Finalize stream and get final transcript.

        Returns:
            Final complete transcript

        """
        try:
            if self.connection:
                # For streaming ASR, prioritize low latency over final confirmation
                # Wait only briefly for final transcript, then use partial
                WAITING_TIME_SEC = 0.4
                if self.partial_transcript and not self.is_final_received:
                    logger.debug("Have partial transcript, waiting briefly for final...")
                    try:
                        await asyncio.wait_for(
                            self._wait_for_final(),
                            timeout=WAITING_TIME_SEC,  # Very short wait - latency is key!
                        )
                        logger.debug("Received final transcript")
                    except asyncio.TimeoutError:
                        logger.debug(f"Using partial transcript (final not received within {1000*WAITING_TIME_SEC}ms)")

                # Get transcript now (don't wait for connection cleanup)
                transcript = self.final_transcript or self.partial_transcript

                # Close connection in background - don't block on cleanup!
                # We already have the transcript, no need to wait
                connection_to_close = self.connection
                self.connection = None

                # Fire-and-forget cleanup task
                asyncio.create_task(self._cleanup_connection(connection_to_close))

                logger.info(f"Deepgram transcript: '{transcript}' (final: {bool(self.is_final_received)})")
                return transcript.strip()
            else:
                # No connection, return what we have
                transcript = self.final_transcript or self.partial_transcript
                logger.info(f"Deepgram transcript: '{transcript}' (final: {bool(self.is_final_received)})")
                return transcript.strip()

        except Exception as e:
            logger.exception(f"Error ending Deepgram stream: {e}")
            # Return what we have
            return (self.final_transcript or self.partial_transcript).strip()

    async def _cleanup_connection(self, connection: Any) -> None:
        """Clean up connection in background (non-blocking)."""
        try:
            await connection.finish()
            logger.debug("Deepgram connection closed")
        except Exception as e:
            logger.debug(f"Connection cleanup error (non-critical): {e}")

    async def _wait_for_final(self) -> None:
        """Wait for final transcript to be received."""
        while not self.is_final_received:
            await asyncio.sleep(0.1)

    async def _on_message(self, *args: Any, **kwargs: Any) -> None:
        """Handle transcript message from Deepgram."""
        try:
            # Extract result from args
            result = kwargs.get("result") or (args[1] if len(args) > 1 else args[0])

            # Parse transcript
            if result and hasattr(result, "channel"):
                channel = result.channel
                if channel and hasattr(channel, "alternatives") and channel.alternatives:
                    alternative = channel.alternatives[0]
                    transcript = alternative.transcript

                    if transcript:
                        # Check if this is a final transcript
                        is_final = result.is_final if hasattr(result, "is_final") else False

                        if is_final:
                            logger.debug(f"Deepgram final: '{transcript}'")
                            self.final_transcript += " " + transcript
                            self.is_final_received = True
                        else:
                            logger.debug(f"Deepgram partial: '{transcript}'")
                            self.partial_transcript = transcript

        except Exception as e:
            logger.warning(f"Error handling Deepgram message: {e}")

    async def _on_error(self, *args: Any, **kwargs: Any) -> None:
        """Handle error from Deepgram."""
        error = kwargs.get("error") or (args[1] if len(args) > 1 else args[0])
        logger.error(f"Deepgram error: {error}")

    def _wav_to_pcm(self, audio_bytes: bytes) -> bytes:
        """Convert WAV audio to raw PCM int16.

        Args:
            audio_bytes: WAV file bytes

        Returns:
            Raw PCM audio data (int16)

        """
        try:
            # Try to parse as WAV
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                # Read frames (this is raw PCM)
                pcm_data = wav_file.readframes(wav_file.getnframes())
                return pcm_data
        except Exception:
            # Assume it's already raw PCM
            return audio_bytes
