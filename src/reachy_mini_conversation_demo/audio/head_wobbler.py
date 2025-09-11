import asyncio
import base64

import numpy as np
from pyparsing import Optional

from reachy_mini_conversation_demo.audio.speech_tapper import HOP_MS, SwayRollRT

SAMPLE_RATE = 24000
MOVEMENT_LATENCY_S = 0.08  # seconds between audio and robot movement


class HeadWobbler:
    def __init__(self, set_offsets):
        self.set_offsets = set_offsets
        self._base_ts = None
        self._hops_done: int = 0

        self.audio_queue = asyncio.Queue()
        self.sway = SwayRollRT()
        self._update_event = asyncio.Event()  # signals new offsets
        self.task = asyncio.create_task(self.consumer())

    def feed(self, delta_b64):
        buf = np.frombuffer(base64.b64decode(delta_b64), dtype=np.int16).reshape(1, -1)
        self.audio_queue.put_nowait((SAMPLE_RATE, buf))

    async def consumer(self):
        """Convert streaming audio chunks into head-offset poses at precise times."""
        hop_dt = HOP_MS / 1000.0
        loop = asyncio.get_running_loop()

        while True:
            sr, chunk = await self.audio_queue.get()  # (1,N), int16
            pcm = np.asarray(chunk).squeeze(0)
            results = self.sway.feed(pcm, sr)  # list of dicts with keys x_mm..yaw_rad

            if self._base_ts is None:
                # anchor when first audio samples of this utterance arrive
                self._base_ts = loop.time()

            i = 0
            while i < len(results):
                if self._base_ts is None:
                    self._base_ts = loop.time()
                    continue

                target = self._base_ts + MOVEMENT_LATENCY_S + self._hops_done * hop_dt
                now = loop.time()

                # if late by ≥1 hop, drop poses to catch up (no drift accumulation)
                if now - target >= hop_dt:
                    lag_hops = int((now - target) / hop_dt)
                    drop = min(
                        lag_hops, len(results) - i - 1
                    )  # keep at least one to show
                    if drop > 0:
                        self._hops_done += drop
                        i += drop
                        continue

                # if early, wait
                if target > now:
                    await asyncio.sleep(target - now)

                r = results[i]
                # meters + radians
                offsets = (
                    r["x_mm"] / 1000.0,
                    r["y_mm"] / 1000.0,
                    r["z_mm"] / 1000.0,
                    r["roll_rad"],
                    r["pitch_rad"],
                    r["yaw_rad"],
                )
                self.set_offsets(offsets)
                self._hops_done += 1
                i += 1
                self._update_event.set()  # notify waiters of a fresh value
