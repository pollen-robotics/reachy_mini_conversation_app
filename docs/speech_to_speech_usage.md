---
title: "Give Reachy Mini your own voice: running a local Speech-to-Speech engine"
thumbnail: /blog/assets/reachy_mini_s2s/thumbnail.png
authors:
- user: A-Mahla
- user: andito
---

# Give Reachy Mini your own voice: running a local Speech-to-Speech engine

The [Reachy Mini conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app) ships with a hosted Hugging Face realtime backend, so you can plug in a robot and start talking out of the box. But the whole point of an open-source robot is that you get to choose what runs inside it. In this post we walk through pairing Reachy Mini with our open-source [`speech-to-speech`](https://github.com/huggingface/speech-to-speech) engine, a cascaded VAD → STT → LLM → TTS pipeline that exposes an Responses API-compatible `/v1/realtime` WebSocket, and we point the robot at it through two lines of `.env`.

We will be opinionated about the pieces that should *just work* (VAD, STT, TTS) and open about the piece where you actually want to experiment: the LLM.

> **TL;DR**
> - Defaults we recommend: **Silero VAD**, **Parakeet-TDT STT**, **Qwen3-TTS**.
> - For the LLM, start local with **MLX** or **Transformers** running **Qwen3-4B-Instruct-2507**.
> - Need more flexibility? Plug any **Responses API** server: vLLM, llama.cpp, a Hugging Face Inference Endpoint, or OpenAI.
> - Point Reachy Mini at your engine with:
>   ```env
>   HF_REALTIME_CONNECTION_MODE="local"
>   HF_REALTIME_WS_URL="ws://127.0.0.1:8765/v1/realtime"
>   ```

---

## Why run your own Speech-to-Speech server?

Hosted realtime backends are convenient, but they come with three trade-offs that matter for a robot sitting on your desk:

- **Latency.** Audio round-trips to a cloud endpoint add a few hundred milliseconds to every turn. A local engine on the same Wi-Fi (or the same laptop) feels noticeably snappier.
- **Privacy.** Audio never leaves your machine.
- **Choice of brain.** With your own engine you can swap the LLM at will. A small MLX model for a quick demo, a vLLM-served 70B for a serious agent, or a hosted frontier model behind a Responses API for the hard prompts.

The `speech-to-speech` repo gives you all of that in a single CLI. It boots a WebSocket server at `/v1/realtime` that speaks the same protocol Reachy Mini already knows how to talk to.

## Install the engine

```bash
git clone https://github.com/huggingface/speech-to-speech.git
cd speech-to-speech
uv sync
```

That gives you the `speech-to-speech` entrypoint. Add `[mlx-lm]`, `[paraformer]`, `[faster-whisper]`, etc. as extras if you want to swap any of the default backends later.

## Our opinionated defaults: VAD, STT, TTS

A voice pipeline has four moving parts. Three of them — *when the user is speaking*, *what they said*, and *how the robot says its answer* — are problems where good defaults exist and where you should not have to tune anything to get a great experience. We recommend the same trio we ship in production:

| Stage | Choice | Why |
|-------|--------|-----|
| VAD | **Silero VAD v5** | Tiny, accurate, runs on CPU. The de-facto default in the open-source voice-agent world. |
| STT | **Parakeet-TDT** | Streaming-friendly, very fast, great quality on English. |
| TTS | **Qwen3-TTS** | Expressive, low-latency, multilingual, supports custom voices. |

These match the defaults in the `speech-to-speech` repo, so the only thing you actually need to choose is the LLM.

## Choosing your LLM

This is where you have decisions to make. We group the options into two families: **run a model inside the engine** (MLX or Transformers), or **let the engine talk to a separate inference server over a Responses API** (vLLM, llama.cpp, HF Inference Endpoints, OpenAI).

### Option 1 — Local LLM on MLX (Apple Silicon)

If you are on a Mac, MLX is the lowest-friction way to run a real model with sane latency. We recommend **Qwen3-4B-Instruct-2507**, which is small enough to feel instant on M-series chips and capable enough to hold a conversation.

```bash
speech-to-speech \
  --mode realtime \
  --stt parakeet-tdt \
  --tts qwen3 \
  --llm_backend mlx-lm \
  --model_name "mlx-community/Qwen3-4B-Instruct-2507-bf16"
```

The server listens on `ws://127.0.0.1:8765/v1/realtime` by default. Leave it running, jump to the [Connecting Reachy Mini](#connect-reachy-mini-to-your-engine) section, and you are talking to your robot.

### Option 2 — Local LLM on Transformers (CUDA / CPU / MPS)

Same idea, but using vanilla `transformers`. Use this if you are on a CUDA box, on Linux, or if you want to swap models freely without re-converting weights for MLX.

```bash
speech-to-speech \
  --mode realtime \
  --stt parakeet-tdt \
  --tts qwen3 \
  --llm_backend transformers \
  --model_name "Qwen/Qwen3-4B-Instruct-2507"
```

> **Tip.** `Qwen3-4B-Instruct-2507` is our default recommendation because it gives a good speed/quality balance on a single consumer GPU. You can point `--model_name` at any HF model the backend supports — for example a larger Qwen, a Llama, or a Mistral.

### The Responses API: decouple the brain from the voice loop

The two options above bundle the LLM inside the `speech-to-speech` process. That is convenient, but it has a downside: every time you restart the voice loop, you reload the LLM weights. And you cannot easily share the same model with other apps.

The `speech-to-speech` engine therefore supports a second mode where the LLM lives in a separate process — any process — as long as it speaks the OpenAI Responses API protocol. You launch your model server in one terminal, you launch the voice loop in another terminal, and the two talk over HTTP.

This is the layout that scales: you keep the heavy weights warm in their own server, and the voice loop becomes a thin client you can restart at will.

#### Option 3 — vLLM in one terminal, speech-to-speech in the other

**Terminal 1 — vLLM inference server:**

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --port 8000 \
  --host 127.0.0.1
```

**Terminal 2 — speech-to-speech client:**

```bash
speech-to-speech \
  --mode realtime \
  --stt parakeet-tdt \
  --tts qwen3 \
  --llm_backend responses-api \
  --model_name "Qwen/Qwen3-4B-Instruct-2507" \
  --responses_api_base_url "http://127.0.0.1:8000/v1"
```

#### Option 4 — llama.cpp in one terminal, speech-to-speech in the other

If you prefer GGUF weights or are running on a machine where vLLM is awkward, `llama-server` gives you the same OpenAI-compatible endpoint.

**Terminal 1 — llama.cpp server:**

```bash
llama-server \
  -hf bartowski/Qwen3-4B-Instruct-2507-GGUF \
  --port 8000 \
  --host 127.0.0.1
```

**Terminal 2 — speech-to-speech client:**

```bash
speech-to-speech \
  --mode realtime \
  --stt parakeet-tdt \
  --tts qwen3 \
  --llm_backend responses-api \
  --model_name "qwen3-4b-instruct" \
  --responses_api_base_url "http://127.0.0.1:8000/v1"
```

#### Option 5 — Hugging Face Inference Endpoints

Same protocol, but the model runs on a managed GPU on Hugging Face. Deploy any chat model as an Inference Endpoint, then point the voice loop at the endpoint URL:

```bash
speech-to-speech \
  --mode realtime \
  --stt parakeet-tdt \
  --tts qwen3 \
  --llm_backend responses-api \
  --model_name "Qwen/Qwen3-4B-Instruct-2507" \
  --responses_api_base_url "https://<your-endpoint>.endpoints.huggingface.cloud/v1" \
  --responses_api_api_key "$HF_TOKEN"
```

#### Option 6 — OpenAI (or any OpenAI-compatible provider)

When you want to test against a frontier model with zero infra, point the same flag at OpenAI:

```bash
speech-to-speech \
  --mode realtime \
  --stt parakeet-tdt \
  --tts qwen3 \
  --llm_backend responses-api \
  --model_name "gpt-4o-mini" \
  --responses_api_api_key "$OPENAI_API_KEY"
```

The `--responses_api_*` flags work the same for any provider that implements the protocol (OpenRouter, Together, Fireworks, …). Swap the base URL and the API key, keep the rest of the pipeline identical.

---

## Connect Reachy Mini to your engine

Once `speech-to-speech` is running and printing something like `Realtime server listening on ws://127.0.0.1:8765/v1/realtime`, all that is left is to tell the conversation app to talk to it instead of the hosted backend.

In the root of the `reachy_mini_conversation_app` checkout, copy `.env.example` to `.env` and set:

```env
BACKEND_PROVIDER="huggingface"
HF_REALTIME_CONNECTION_MODE="local"
HF_REALTIME_WS_URL="ws://127.0.0.1:8765/v1/realtime"
```

Then launch the app as usual:

```bash
reachy-mini-conversation-app
```

That is the entire integration. The `HF_REALTIME_CONNECTION_MODE="local"` flag flips the app from the hosted Space proxy to a direct WebSocket; `HF_REALTIME_WS_URL` says *where* that WebSocket lives.

### Running the engine on your laptop, the app on the robot

If you are running the voice engine on your laptop and the conversation app on a Reachy Mini Wireless, the only thing that changes is the URL — make sure the engine binds to a LAN address (not just `127.0.0.1`) and use the laptop's IP from the robot:

```env
BACKEND_PROVIDER="huggingface"
HF_REALTIME_CONNECTION_MODE="local"
HF_REALTIME_WS_URL="ws://<your-laptop-lan-ip>:8765/v1/realtime"
```

If the engine has to stay bound to loopback for any reason, an SSH reverse tunnel from your laptop into the robot works just as well:

```bash
ssh -N -R 8765:127.0.0.1:8765 <robot-user>@<robot-host>
```

…and then keep the original loopback URL on the robot.

---

## Wrap up

You now have a fully local voice loop:

- A robot listening with **Silero**,
- transcribing with **Parakeet-TDT**,
- thinking with whichever LLM you picked — local MLX, local Transformers, a vLLM/llama.cpp server next door, or a hosted Responses API endpoint,
- and answering with **Qwen3-TTS**.

The defaults are opinionated on purpose: VAD/STT/TTS are solved problems and we want them out of your way. The LLM is the part of the stack that should keep changing as the open models keep getting better — and the Responses API is the seam that lets you swap it without ever touching the voice loop.

Star [`huggingface/speech-to-speech`](https://github.com/huggingface/speech-to-speech) and [`pollen-robotics/reachy_mini_conversation_app`](https://github.com/pollen-robotics/reachy_mini_conversation_app), and come tell us in the discussions which LLM you ended up running on your robot.
