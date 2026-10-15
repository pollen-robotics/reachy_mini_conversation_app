import hashlib
from pathlib import Path
from functools import partial
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

import reachy_mini_conversation_app.audio.local_vad as vad_mod
from reachy_mini_conversation_app.audio.local_vad import SileroVAD


@pytest.fixture
def model_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, MagicMock]:
    """Isolate the model cache and upstream downloads from the network."""
    responses = {
        "src/silero_vad/data/silero_vad_16k_op15.onnx": b"test-model",
        "LICENSE": b"test-license",
    }

    def respond(request: httpx.Request) -> httpx.Response:
        prefix = "https://raw.githubusercontent.com/snakers4/silero-vad/7e30209a3e901f9842f81b225f3e93d8199902b1/"
        assert str(request.url).startswith(prefix)
        return httpx.Response(200, content=responses[str(request.url).removeprefix(prefix)])

    download = MagicMock(side_effect=respond)
    monkeypatch.setattr(vad_mod, "cached_assets_path", lambda *_args: tmp_path)
    monkeypatch.setattr(
        vad_mod,
        "_MODEL_FILES",
        {
            name: (upstream_path, hashlib.sha256(responses[upstream_path]).hexdigest())
            for name, (upstream_path, _checksum) in vad_mod._MODEL_FILES.items()
        },
    )
    monkeypatch.setattr(vad_mod.httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(download)))
    monkeypatch.setattr(vad_mod.ort, "InferenceSession", MagicMock())
    return tmp_path, download


def test_streaming_speech_holds_through_pauses_and_resets(
    monkeypatch: pytest.MonkeyPatch, model_cache: tuple[Path, MagicMock]
) -> None:
    """Partial frames retain model context, and brief pauses do not end listening."""
    session = MagicMock()
    probabilities = iter([*([0.8] * 5), *([0.1] * 12), 0.8, *([0.1] * 13), *([0.8] * 5), 0.1])

    def infer(_outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.array([[next(probabilities)]], dtype=np.float32), np.ones_like(inputs["state"])]

    session.run.side_effect = infer
    monkeypatch.setattr(vad_mod.ort, "InferenceSession", lambda *_args, **_kwargs: session)
    detector = SileroVAD()
    half_chunk = np.full(256, 16384, dtype=np.int16)
    silence = np.zeros(512, dtype=np.int16)

    assert not detector.process(half_chunk, 16000)
    session.run.assert_not_called()
    assert not detector.process(half_chunk, 16000)
    assert detector.speech_probability == pytest.approx(0.8)
    first_inputs = session.run.call_args.args[1]
    np.testing.assert_array_equal(first_inputs["input"][:, :64], 0.0)
    np.testing.assert_array_equal(first_inputs["input"][:, 64:], 0.5)
    for _ in range(3):
        assert not detector.process(silence, 16000)
    assert detector.process(silence, 16000)

    for _ in range(12):
        assert detector.process(silence.reshape(-1, 1), 16000)
    assert detector.process(silence, 16000)
    for _ in range(12):
        assert detector.process(silence, 16000)
    assert not detector.process(silence, 16000)

    for _ in range(4):
        assert not detector.process(silence, 16000)
    assert detector.process(silence, 16000)
    detector.reset()
    assert detector.speech_probability == 0.0
    assert not detector.process(half_chunk, 16000)
    assert not detector.process(half_chunk, 16000)
    reset_inputs = session.run.call_args.args[1]
    np.testing.assert_array_equal(reset_inputs["state"], 0.0)
    np.testing.assert_array_equal(reset_inputs["input"][:, :64], 0.0)

    with pytest.raises(ValueError, match="16 kHz"):
        detector.process(silence, 48000)


def test_first_start_downloads_model_and_license_then_reuses_cache_offline(
    model_cache: tuple[Path, MagicMock],
) -> None:
    """First use downloads both files; later starts work without a connection."""
    cache, download = model_cache
    SileroVAD()
    assert download.call_count == 2
    assert (cache / "silero_vad_16k_op15.onnx").read_bytes() == b"test-model"
    assert (cache / "LICENSE").read_bytes() == b"test-license"
    download.reset_mock()
    download.side_effect = httpx.ConnectError("offline")
    SileroVAD()
    download.assert_not_called()
    assert {path.name for path in cache.iterdir()} == {"silero_vad_16k_op15.onnx", "LICENSE"}


def test_corrupted_cached_model_is_replaced(
    model_cache: tuple[Path, MagicMock], caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupted model is downloaded again before inference starts."""
    cache, download = model_cache
    SileroVAD()
    download.reset_mock()
    (cache / "silero_vad_16k_op15.onnx").write_bytes(b"corrupted")
    SileroVAD()
    assert download.call_count == 1
    assert (cache / "silero_vad_16k_op15.onnx").read_bytes() == b"test-model"
    assert "Cached Silero file failed checksum verification" in caplog.text


@pytest.mark.parametrize("failure", ["offline", "checksum", "license"])
def test_failed_download_does_not_load_unverified_model(model_cache: tuple[Path, MagicMock], failure: str) -> None:
    """Download failures leave no incomplete files and never start inference."""
    cache, download = model_cache
    if failure == "offline":
        download.side_effect = httpx.ConnectError("offline")
        expected_error = httpx.ConnectError
    elif failure == "checksum":
        download.side_effect = lambda request: httpx.Response(200, content=b"wrong-model")
        expected_error = ValueError
    else:
        download.side_effect = [
            httpx.Response(200, content=b"test-model"),
            httpx.Response(404),
        ]
        expected_error = httpx.HTTPStatusError
    with pytest.raises(expected_error):
        SileroVAD()
    vad_mod.ort.InferenceSession.assert_not_called()
    assert not (cache / "LICENSE").exists()
    assert {path.name for path in cache.iterdir()} <= {"silero_vad_16k_op15.onnx"}


@pytest.mark.parametrize("batched", [False, True])
def test_barge_in_requires_active_speech_and_rejects_noise(
    monkeypatch: pytest.MonkeyPatch, model_cache: tuple[Path, MagicMock], batched: bool
) -> None:
    """Movement's silence hold cannot confirm an interruption after a single noisy frame."""
    session = MagicMock()
    probabilities = iter([0.7, 0.1, 0.1, 0.7, *([0.5] * 11), 0.1, 0.1, *([0.8] * 12)])
    session.run.side_effect = lambda *_args: [
        np.array([[next(probabilities)]], dtype=np.float32),
        np.zeros((2, 1, 128), dtype=np.float32),
    ]
    monkeypatch.setattr(vad_mod.ort, "InferenceSession", lambda *_args, **_kwargs: session)
    detector = SileroVAD()
    chunk = np.zeros(512, dtype=np.int16)
    detector.process(chunk, 16000, detect_barge_in=True)
    assert detector.barge_in_candidate and not detector.barge_in_confirmed
    for _ in range(2):
        detector.process(chunk, 16000, detect_barge_in=True)
    assert not detector.barge_in_candidate and not detector.barge_in_confirmed
    if batched:
        detector.process(np.zeros(512 * 12, dtype=np.int16), 16000, detect_barge_in=True)
    else:
        for _ in range(11):
            detector.process(chunk, 16000, detect_barge_in=True)
            assert not detector.barge_in_confirmed
        detector.process(chunk, 16000, detect_barge_in=True)
    assert detector.barge_in_confirmed
    for _ in range(2):
        detector.process(chunk, 16000, detect_barge_in=True)
    assert not detector.barge_in_candidate
    for _ in range(12):
        detector.process(chunk, 16000)
    assert not detector.barge_in_candidate and not detector.barge_in_confirmed
    detector.reset()
    assert not detector.barge_in_confirmed


def test_batched_speech_followed_by_silence_still_confirms_barge_in(
    monkeypatch: pytest.MonkeyPatch, model_cache: tuple[Path, MagicMock]
) -> None:
    """A completed valid utterance inside a large microphone buffer still interrupts."""
    probabilities = iter([*([0.8] * 12), 0.1, 0.1])
    session = MagicMock()
    session.run.side_effect = lambda *_args: [
        np.array([[next(probabilities)]], dtype=np.float32),
        np.zeros((2, 1, 128), dtype=np.float32),
    ]
    monkeypatch.setattr(vad_mod.ort, "InferenceSession", lambda *_args, **_kwargs: session)
    detector = SileroVAD()
    detector.process(np.zeros(512 * 14, dtype=np.int16), 16000, detect_barge_in=True)
    assert not detector.barge_in_candidate
    assert detector.barge_in_confirmed


@pytest.mark.parametrize("input_samples", [173, 512, 4096])
def test_model_resets_preserve_audio_and_context_during_speech(
    monkeypatch: pytest.MonkeyPatch, model_cache: tuple[Path, MagicMock], input_samples: int
) -> None:
    """Idle and segment resets preserve every PCM sample and context throughout active speech and pauses."""
    probabilities = iter([*([0.1] * 16), *([0.8] * 5), *([0.1] * 12), 0.8, *([0.1] * 13), *([0.8] * 5), 0.1, 0.1])
    chunks: list[np.ndarray] = []
    states: list[np.ndarray] = []
    contexts: list[np.ndarray] = []

    def infer(_outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        chunks.append(inputs["input"][:, 64:].copy())
        states.append(inputs["state"].copy())
        contexts.append(inputs["input"][:, :64].copy())
        return [np.array([[next(probabilities)]], dtype=np.float32), np.ones_like(inputs["state"])]

    session = MagicMock()
    session.run.side_effect = infer
    monkeypatch.setattr(vad_mod.ort, "InferenceSession", lambda *_args, **_kwargs: session)
    detector = SileroVAD()
    audio = np.arange(53 * 512 + 137, dtype=np.int16)
    listening = [
        detector.process(audio[offset : offset + input_samples], 16000)
        for offset in range(0, audio.size, input_samples)
    ]
    assert listening[-1]
    assert session.run.call_count == 53
    assert detector.process(np.zeros(375, dtype=np.int16), 16000)

    expected = np.concatenate((audio, np.zeros(375, dtype=np.int16))).astype(np.float32) / 32768.0
    np.testing.assert_array_equal(np.concatenate(chunks, axis=1).ravel(), expected)
    for index, (state, context) in enumerate(zip(states, contexts)):
        if index in (0, 16, 47):
            np.testing.assert_array_equal(state, 0.0)
            np.testing.assert_array_equal(context, 0.0)
        else:
            np.testing.assert_array_equal(state, 1.0)
            np.testing.assert_array_equal(context, chunks[index - 1][:, -64:])

    if input_samples == 512:
        assert not any(listening[:20])
        assert all(listening[20:46])
        assert not any(listening[46:51])
        assert all(listening[51:])


def test_short_bursts_are_ignored_and_uncertain_audio_counts_as_silence(
    monkeypatch: pytest.MonkeyPatch, model_cache: tuple[Path, MagicMock]
) -> None:
    """Knocks under 160 ms never vote, and hum below the speech threshold still ends listening."""
    probabilities = iter([*([0.8] * 4), 0.1, *([0.8] * 5), *([0.4] * 30)])
    states: list[np.ndarray] = []

    def infer(_outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        states.append(inputs["state"].copy())
        return [np.array([[next(probabilities)]], dtype=np.float32), np.ones_like(inputs["state"])]

    session = MagicMock()
    session.run.side_effect = infer
    monkeypatch.setattr(vad_mod.ort, "InferenceSession", lambda *_args, **_kwargs: session)
    detector = SileroVAD()
    frame = np.full(512, 16384, dtype=np.int16)
    listening = [detector.process(frame, 16000) for _ in range(40)]

    assert not any(listening[:9])
    assert all(listening[9:22])
    assert not any(listening[22:])
    assert [index for index, state in enumerate(states) if not np.any(state)] == [0, 23, 39]
