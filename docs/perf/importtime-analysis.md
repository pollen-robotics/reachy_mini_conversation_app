# Import-time profile analysis (boot perf, issue #277)

**Profile date:** 2026-05-14
**Host:** `reachy-mini` (Pi 5, on-robot chassis) via `ricci` ssh alias
**Command:** `timeout 25 /venvs/apps_venv/bin/python -X importtime -u -m robot_comic.main`
**Backend selected at boot (per app log embedded in profile):** `BACKEND_PROVIDER=local_stt`, audio I/O pair `moonshine` / `elevenlabs`
**Raw profile:** [`importtime-2026-05-14.txt`](./importtime-2026-05-14.txt) (3071 lines; truncated at the 25s watchdog while `scipy.signal._spline_filters` was still loading from Moonshine init)
**Parsed summary:** [`importtime-parsed-summary.txt`](./importtime-parsed-summary.txt)

> Note: because the run is timeout-clipped, the *cumulative* numbers for
> packages still in flight (most notably anything inside Moonshine /
> `usefulsensors_moonshine_onnx` and the FastRTC TURN/aiortc warmup) are
> lower-bound. Top-line `self` numbers are still accurate for what we
> captured.

## Top 20 imports by `self` time

| rank | self (ms) | cum (ms) | module |
| ---: | ---: | ---: | --- |
|  1 | 5516.5 | 5804.8 | `google.genai.types` |
|  2 |  753.9 |  753.9 | `gradio.templates` |
|  3 |  448.0 |  468.9 | `fastapi.openapi.models` |
|  4 |  323.6 |  409.1 | `scipy.ndimage._support_alternative_backends` |
|  5 |  287.1 |  287.1 | `pyparsing.core` |
|  6 |  275.7 |  378.9 | `gradio.components.multimodal_textbox` |
|  7 |  251.6 |  739.6 | `gradio_client.client` |
|  8 |  189.2 |  555.9 | `scipy.signal.windows._windows` |
|  9 |  176.6 |  176.6 | `gradio.components.native_plot` |
| 10 |  176.5 |  176.5 | `gradio.components.chatbot` |
| 11 |  166.8 |  170.2 | `cv2` |
| 12 |  166.0 |  873.8 | `gradio.components.base` |
| 13 |  148.2 |  148.2 | `gradio.components.audio` |
| 14 |  146.1 |  150.1 | `gradio.components.video` |
| 15 |  142.8 |  156.9 | `charset_normalizer.api` |
| 16 |  137.6 |  137.6 | `gradio.components.image_editor` |
| 17 |  132.4 |  133.5 | `sounddevice` |
| 18 |  131.9 |  132.4 | `scipy.special._support_alternative_backends` |
| 19 |  125.9 |  142.8 | `scipy.fft._basic` |
| 20 |  123.2 |  165.2 | `matplotlib.patches` |

## Top 10 packages by cumulative time

The cumulative number is the sum of `self` plus all transitive imports
triggered by the package's top-level statements.

| rank | self (ms) | cum (ms) | top-level package |
| ---: | ---: | ---: | --- |
|  1 |   3.1 | **8045.9** | `fastrtc` (pulls gradio + aiortc + google-genai) |
|  2 |   1.5 | **7143.3** | `google.genai` (chiefly its giant `types` module) |
|  3 |   0.1 | 6740.6 | `gradio.data_classes` (proxy for the gradio chain) |
|  4 |   4.6 | 2886.7 | `reachy_mini` |
|  5 |   1.0 | 2739.6 | `reachy_mini_toolbox.vision` (loads mediapipe) |
|  6 |   0.9 | 2736.3 | `mediapipe` |
|  7 |  36.2 | 1781.4 | `matplotlib.pyplot` (pulled by gradio) |
|  8 |   3.6 | 1602.2 | `scipy.signal._spline_filters` (Moonshine) |
|  9 |   0.8 | 1099.2 | `fastapi` (pulled by fastrtc) |
| 10 |   1.0 | 740.3 | `gradio_client.utils` |

## Sum of `self` time grouped by top-level third-party package

| self (ms) | count | package |
| ---: | ---: | --- |
| 6716.5 | 241 | google (genai SDK) |
| 5025.8 | 145 | gradio |
| 2852.6 | 404 | scipy |
| 1163.2 |  94 | matplotlib |
|  619.9 | 103 | huggingface_hub |
|  598.0 |  41 | fastapi |
|  514.7 | 241 | mediapipe |
|  455.1 | 147 | numpy |
|  363.7 |  11 | pyparsing |
|  296.8 |  12 | gradio_client |
|  260.6 |  70 | opentelemetry |
|  249.1 |  58 | av |
|  231.0 |  74 | PIL |
|  226.5 |  40 | aiohttp |
|  203.7 |  33 | robot_comic |
|  187.8 |  34 | reachy_mini |
|  181.6 |  17 | fastrtc |
|  178.9 |  44 | pydantic |
|  178.4 |  31 | aiortc |
|  169.8 |   5 | cv2 |

## Imports ≥ 200 ms `self` and recommendations

The threshold from issue #277 is "≥ 200 ms self → decide lazy / drop / accept".
Only seven entries clear that bar. They are all transitively pulled by
**`elevenlabs_tts.py`** (the chosen output backend), via two distinct chains:
the `google.genai` SDK and the `fastrtc → gradio` chain.

### 1. `google.genai.types` — 5516 ms self / 5805 ms cum

**Pulled by:** `from google.genai import types` at the top of
`src/robot_comic/elevenlabs_tts.py:22` (also `gemini_tts.py:22`,
`gemini_live.py`, `gemini_llm.py`).
**Reason for cost:** `google.genai.types` eagerly constructs the full
pydantic schema for the entire genai REST surface (`google.genai._interactions.types.*` — hundreds of submodules, each defining a `BaseModel`).
**Recommendation:** **Lazy-load.** This single import is more than a sixth
of the entire boot budget. The handler classes only need `types` inside
methods that actually construct a request (e.g. `_build_contents`,
`_make_generate_call`). Move both `from google import genai` and
`from google.genai import types` into the first function/method that
needs them, or into a memoized helper. Same fix in all four files that do
the eager import.
**Expected savings:** ~5.5s off cold-boot for any backend that selects an
elevenlabs/gemini path. Recovered cost is paid once on first turn, when
the user is already speaking — invisible.

### 2. `gradio.templates` — 754 ms self

**Pulled by:** `fastrtc` package top-level → `gradio` → `gradio.templates`.
`elevenlabs_tts.py:21` does `from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item`. fastrtc’s `__init__` eagerly imports
`Stream`, which eagerly imports gradio.
**Recommendation:** **Lazy-load the fastrtc import surface in handler
modules.** The handler classes need `AsyncStreamHandler` for inheritance
(class statement runs at import time), so we cannot trivially defer the
base class. Two viable options:
1. Split `AdditionalOutputs` / `wait_for_item` (functions, used in
   methods) into a local import inside the methods that use them; keep
   only `AsyncStreamHandler` at module top. This won't drop fastrtc
   itself but cuts the import surface.
2. Larger lever: ask upstream fastrtc to make `Stream` (which is the
   gradio entry point) lazy. Until that lands, monkey-patch /
   conditionally import `fastrtc.Stream` only in sim mode.
**Expected savings:** the whole `gradio` cumulative budget is **~6.7s**.
Realistic recovery without an upstream change: ~750 ms (the gradio
templates eager load) + ~440 ms (fastapi.openapi.models, see below) if
fastrtc's webrtc subpath can be deferred.

### 3. `fastapi.openapi.models` — 448 ms self / 469 ms cum

**Pulled by:** `fastrtc → fastapi → fastapi.exceptions → fastapi.openapi.models`.
**Reason for cost:** OpenAPI schema models are defined as a huge pydantic
class graph. The admin UI is FastAPI, so we do need fastapi in the
process — but only after `run()` constructs `console.LocalStream` and
mounts routes, well after handler-ready.
**Recommendation:** **Accept for now**, but bundle with #2 — if fastrtc
import is moved behind a sim-mode-only branch (already done in
`main.py:392-396`), then `fastapi` would still come in via
`headless_personality_ui` and `console`. Those are also already
imported lazily inside `run()`. The 448 ms hit is therefore *probably*
attributable to fastrtc's eager import in handler modules, not console.
Will fold into the fastrtc ticket.

### 4. `scipy.ndimage._support_alternative_backends` — 324 ms self / 409 ms cum

**Pulled by:** `gradio.components.image_editor → scipy.ndimage`.
**Recommendation:** **Wait for #2.** Goes away if gradio import is
deferred. No direct robot_comic import site to attack.

### 5. `pyparsing.core` — 287 ms self

**Pulled by:** `matplotlib → pyparsing` (matplotlib pulled by gradio).
**Recommendation:** **Wait for #2.** Goes away with gradio.

### 6. `gradio.components.multimodal_textbox` — 276 ms self

**Pulled by:** gradio. **Recommendation:** wait for #2.

### 7. `gradio_client.client` — 252 ms self / 740 ms cum

**Pulled by:** gradio. **Recommendation:** wait for #2.

## Other modules worth a note (sub-200ms but suggestive)

- `cv2` (167 ms self, 170 ms cum) is pulled by `mediapipe` and
  by `reachy_mini.media.camera_utils`. The mediapipe path is already
  lazy via `robot_comic.vision.head_tracking.mediapipe`; the camera-utils
  path is part of `reachy_mini` core and not in our control here.
- `sounddevice` (132 ms self) is imported by `reachy_mini.media.audio_*`.
  Worth confirming we actually use sounddevice on the Pi (vs PortAudio
  via reachy_mini.media.audio_gstreamer). Out of scope for #277.
- `huggingface_hub` totals 620 ms summed self time. Hard to retire — it
  is used directly by Moonshine model download/load. Accept.

## Follow-up tickets

Filed against #277:

- #283 — **lazy-load `google.genai` / `google.genai.types`** (~5.5 s) —
  top-level imports in `elevenlabs_tts.py`, `gemini_tts.py`,
  `gemini_live.py`, `gemini_llm.py`. Move into
  the first method that constructs a request.
- #284 — **defer `fastrtc` import surface** (~0.75-1.2 s recoverable
  today; ~6.7 s if upstream cooperates) — split the `AsyncStreamHandler`
  base class (must stay) from the function imports (can be lazy) in
  every handler module. Investigate whether fastrtc's eager `Stream`
  import can be sidestepped on the Pi.

The remaining ≥200 ms entries are downstream of fastrtc; they should
disappear or shrink dramatically when the fastrtc ticket lands.
