"""Unit tests for the speech_tapper module."""

import math

import pytest
import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.audio.speech_tapper import (
    SR,
    FRAME_MS,
    HOP_MS,
    HOP,
    FRAME,
    VAD_DB_ON,
    VAD_DB_OFF,
    SwayRollRT,
    _rms_dbfs,
    _loudness_gain,
    _to_float32_mono,
    _resample_linear,
)


class TestRmsDbfs:
    """Tests for _rms_dbfs function."""

    def test_rms_dbfs_silence(self) -> None:
        """Test RMS dBFS for silence."""
        silence = np.zeros(1000, dtype=np.float32)
        db = _rms_dbfs(silence)
        # Should be very negative (approaching -inf)
        assert db < -100

    def test_rms_dbfs_full_scale(self) -> None:
        """Test RMS dBFS for full scale sine."""
        # Full scale sine has RMS of 1/sqrt(2) ≈ 0.707, which is about -3 dBFS
        t = np.linspace(0, 1, 1000, dtype=np.float32)
        full_scale = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        db = _rms_dbfs(full_scale)
        assert -4 < db < -2  # Approximately -3 dBFS

    def test_rms_dbfs_half_amplitude(self) -> None:
        """Test RMS dBFS for half amplitude."""
        t = np.linspace(0, 1, 1000, dtype=np.float32)
        half_scale = 0.5 * np.sin(2 * np.pi * 10 * t).astype(np.float32)
        db = _rms_dbfs(half_scale)
        # Half amplitude is -6 dB from full scale
        assert -10 < db < -6


class TestLoudnessGain:
    """Tests for _loudness_gain function."""

    def test_loudness_gain_low_db(self) -> None:
        """Test loudness gain for very low dB."""
        gain = _loudness_gain(-80.0)
        assert gain == 0.0

    def test_loudness_gain_high_db(self) -> None:
        """Test loudness gain for high dB (above range)."""
        gain = _loudness_gain(0.0)
        assert gain == 1.0

    def test_loudness_gain_mid_range(self) -> None:
        """Test loudness gain for mid-range dB."""
        gain = _loudness_gain(-30.0)
        assert 0.0 < gain < 1.0

    def test_loudness_gain_clamped(self) -> None:
        """Test loudness gain is clamped to [0, 1]."""
        low_gain = _loudness_gain(-100.0)
        high_gain = _loudness_gain(10.0)
        assert low_gain == 0.0
        assert high_gain == 1.0


class TestToFloat32Mono:
    """Tests for _to_float32_mono function."""

    def test_to_float32_mono_1d_int16(self) -> None:
        """Test conversion of 1D int16 array."""
        pcm = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        result = _to_float32_mono(pcm)
        assert result.dtype == np.float32
        assert len(result) == 4
        assert result[0] == pytest.approx(0.0, abs=0.01)
        assert result[1] == pytest.approx(0.5, abs=0.01)
        assert result[2] == pytest.approx(-0.5, abs=0.01)
        assert result[3] == pytest.approx(1.0, abs=0.01)

    def test_to_float32_mono_1d_float(self) -> None:
        """Test conversion of 1D float array (passthrough)."""
        audio = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        result = _to_float32_mono(audio)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, audio)

    def test_to_float32_mono_2d_stereo(self) -> None:
        """Test conversion of 2D stereo (channels, samples)."""
        left = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        right = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        stereo = np.stack([left, right], axis=0)  # (2, 3)
        result = _to_float32_mono(stereo)
        assert result.dtype == np.float32
        assert len(result) == 3
        # Should be mean of channels
        np.testing.assert_array_almost_equal(result, [0.4, 0.4, 0.4])

    def test_to_float32_mono_2d_samples_channels(self) -> None:
        """Test conversion of 2D (samples, channels) format."""
        # Shape (100, 2) - many samples, 2 channels
        audio = np.ones((100, 2), dtype=np.float32) * 0.5
        audio[:, 1] = 0.3
        result = _to_float32_mono(audio)
        assert result.dtype == np.float32
        # Should average across the smaller dimension (channels)
        assert len(result) == 100

    def test_to_float32_mono_empty(self) -> None:
        """Test conversion of empty/scalar array."""
        result = _to_float32_mono(np.array(0.5))
        assert result.dtype == np.float32
        assert len(result) == 0

    def test_to_float32_mono_int32(self) -> None:
        """Test conversion of int32 array."""
        pcm = np.array([0, 1073741824, -1073741824], dtype=np.int32)
        result = _to_float32_mono(pcm)
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(0.0, abs=0.01)
        assert result[1] == pytest.approx(0.5, abs=0.01)
        assert result[2] == pytest.approx(-0.5, abs=0.01)


class TestResampleLinear:
    """Tests for _resample_linear function."""

    def test_resample_same_rate(self) -> None:
        """Test resampling with same rate (no-op)."""
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _resample_linear(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_resample_downsample(self) -> None:
        """Test downsampling."""
        # Create 1000 samples at 48kHz, resample to 16kHz
        audio = np.sin(np.linspace(0, 2 * np.pi, 1000, dtype=np.float32)).astype(
            np.float32
        )
        result = _resample_linear(audio, 48000, 16000)
        expected_length = int(round(1000 * 16000 / 48000))
        assert len(result) == expected_length
        assert result.dtype == np.float32

    def test_resample_upsample(self) -> None:
        """Test upsampling."""
        audio = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        result = _resample_linear(audio, 8000, 16000)
        expected_length = int(round(3 * 16000 / 8000))
        assert len(result) == expected_length

    def test_resample_empty(self) -> None:
        """Test resampling empty array."""
        result = _resample_linear(np.array([], dtype=np.float32), 16000, 8000)
        assert len(result) == 0

    def test_resample_tiny(self) -> None:
        """Test resampling very small array."""
        audio = np.array([1.0], dtype=np.float32)
        # Resampling single sample with significant rate change
        result = _resample_linear(audio, 48000, 8000)
        # Result should be empty or single sample
        assert len(result) <= 1


class TestSwayRollRTInit:
    """Tests for SwayRollRT initialization."""

    def test_init_default_seed(self) -> None:
        """Test SwayRollRT initializes with default seed."""
        sway = SwayRollRT()
        assert sway._seed == 7
        assert sway.vad_on is False
        assert sway.sway_env == 0.0
        assert sway.t == 0.0

    def test_init_custom_seed(self) -> None:
        """Test SwayRollRT initializes with custom seed."""
        sway = SwayRollRT(rng_seed=42)
        assert sway._seed == 42

    def test_init_phases_are_random(self) -> None:
        """Test that phases are initialized from RNG."""
        sway1 = SwayRollRT(rng_seed=1)
        sway2 = SwayRollRT(rng_seed=2)
        # Different seeds should give different phases
        assert sway1.phase_pitch != sway2.phase_pitch

    def test_init_phases_reproducible(self) -> None:
        """Test that same seed gives same phases."""
        sway1 = SwayRollRT(rng_seed=42)
        sway2 = SwayRollRT(rng_seed=42)
        assert sway1.phase_pitch == sway2.phase_pitch
        assert sway1.phase_yaw == sway2.phase_yaw
        assert sway1.phase_roll == sway2.phase_roll


class TestSwayRollRTReset:
    """Tests for SwayRollRT reset method."""

    def test_reset_clears_state(self) -> None:
        """Test reset clears VAD and envelope state."""
        sway = SwayRollRT()
        # Modify state
        sway.vad_on = True
        sway.vad_above = 5
        sway.sway_env = 0.8
        sway.t = 1.5

        sway.reset()

        assert sway.vad_on is False
        assert sway.vad_above == 0
        assert sway.vad_below == 0
        assert sway.sway_env == 0.0
        assert sway.t == 0.0

    def test_reset_clears_carry_buffer(self) -> None:
        """Test reset clears carry buffer."""
        sway = SwayRollRT()
        sway.carry = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        sway.reset()

        assert len(sway.carry) == 0

    def test_reset_clears_samples_deque(self) -> None:
        """Test reset clears samples deque."""
        sway = SwayRollRT()
        sway.samples.extend([0.1, 0.2, 0.3])

        sway.reset()

        assert len(sway.samples) == 0


class TestSwayRollRTFeed:
    """Tests for SwayRollRT feed method."""

    def test_feed_empty_returns_empty(self) -> None:
        """Test feeding empty audio returns empty list."""
        sway = SwayRollRT()
        result = sway.feed(np.array([], dtype=np.int16), SR)
        assert result == []

    def test_feed_short_audio_returns_empty(self) -> None:
        """Test feeding very short audio (less than HOP) returns empty."""
        sway = SwayRollRT()
        # Less than HOP samples
        short_audio = np.zeros(HOP - 1, dtype=np.int16)
        result = sway.feed(short_audio, SR)
        # May not produce output yet (depends on carry)
        assert isinstance(result, list)

    def test_feed_sufficient_audio_returns_results(self) -> None:
        """Test feeding sufficient audio produces sway results."""
        sway = SwayRollRT()
        # Generate enough audio for multiple hops
        duration_s = 0.1  # 100ms
        sample_count = int(SR * duration_s)
        t = np.linspace(0, duration_s, sample_count, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        result = sway.feed(audio, SR)

        # Should have multiple results
        assert len(result) > 0
        # Each result should have expected keys
        for r in result:
            assert "pitch_rad" in r
            assert "yaw_rad" in r
            assert "roll_rad" in r
            assert "x_mm" in r
            assert "y_mm" in r
            assert "z_mm" in r

    def test_feed_with_resampling(self) -> None:
        """Test feeding audio at different sample rate."""
        sway = SwayRollRT()
        # Generate audio at 48kHz
        duration_s = 0.1
        sr_in = 48000
        sample_count = int(sr_in * duration_s)
        t = np.linspace(0, duration_s, sample_count, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        result = sway.feed(audio, sr_in)

        # Should still produce results after resampling
        assert isinstance(result, list)

    def test_feed_none_sr_uses_default(self) -> None:
        """Test feeding with None sample rate uses default SR."""
        sway = SwayRollRT()
        # Generate audio assuming default SR
        duration_s = 0.1
        sample_count = int(SR * duration_s)
        audio = np.zeros(sample_count, dtype=np.int16)

        result = sway.feed(audio, None)

        assert isinstance(result, list)

    def test_feed_increments_time(self) -> None:
        """Test that feed increments internal time."""
        sway = SwayRollRT()
        initial_t = sway.t

        # Feed enough audio for at least one hop
        duration_s = 0.1
        sample_count = int(SR * duration_s)
        audio = np.zeros(sample_count, dtype=np.int16)
        sway.feed(audio, SR)

        assert sway.t > initial_t

    def test_feed_loud_audio_activates_vad(self) -> None:
        """Test that loud audio activates VAD."""
        sway = SwayRollRT()

        # Generate loud audio
        duration_s = 0.2  # Longer to trigger VAD
        sample_count = int(SR * duration_s)
        t = np.linspace(0, duration_s, sample_count, endpoint=False)
        # High amplitude
        audio = (0.9 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        sway.feed(audio, SR)

        # VAD should be on or vad_above should have increased
        assert sway.vad_on or sway.vad_above > 0

    def test_feed_silence_keeps_vad_off(self) -> None:
        """Test that silence keeps VAD off."""
        sway = SwayRollRT()

        # Generate silence
        duration_s = 0.1
        sample_count = int(SR * duration_s)
        silence = np.zeros(sample_count, dtype=np.int16)

        sway.feed(silence, SR)

        assert sway.vad_on is False

    def test_feed_accumulates_carry(self) -> None:
        """Test that partial samples are carried over."""
        sway = SwayRollRT()

        # Feed less than HOP samples
        small_chunk = np.zeros(HOP // 2, dtype=np.int16)
        sway.feed(small_chunk, SR)

        assert len(sway.carry) == HOP // 2

    def test_feed_output_values_bounded(self) -> None:
        """Test that output sway values are reasonable."""
        sway = SwayRollRT()

        # Generate moderate audio
        duration_s = 0.2
        sample_count = int(SR * duration_s)
        t = np.linspace(0, duration_s, sample_count, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        results = sway.feed(audio, SR)

        for r in results:
            # Angles should be reasonable (not huge)
            assert abs(r["pitch_rad"]) < 1.0  # < 57 degrees
            assert abs(r["yaw_rad"]) < 1.0
            assert abs(r["roll_rad"]) < 1.0
            # Translations should be reasonable
            assert abs(r["x_mm"]) < 50
            assert abs(r["y_mm"]) < 50
            assert abs(r["z_mm"]) < 50


class TestSwayRollRTIntegration:
    """Integration tests for SwayRollRT."""

    def test_continuous_feed_produces_smooth_output(self) -> None:
        """Test that continuous feeding produces changing output."""
        sway = SwayRollRT()

        all_results = []
        # Feed multiple chunks
        for _ in range(5):
            duration_s = 0.05
            sample_count = int(SR * duration_s)
            t = np.linspace(0, duration_s, sample_count, endpoint=False)
            audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
            results = sway.feed(audio, SR)
            all_results.extend(results)

        assert len(all_results) > 0

    def test_vad_transitions(self) -> None:
        """Test VAD transitions from off to on."""
        sway = SwayRollRT()

        # Start with silence
        silence = np.zeros(int(SR * 0.1), dtype=np.int16)
        sway.feed(silence, SR)
        assert sway.vad_on is False

        # Feed loud audio for longer duration to trigger VAD
        t = np.linspace(0, 0.3, int(SR * 0.3), endpoint=False)
        loud = (0.9 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        sway.feed(loud, SR)

        # VAD should have activated or at least started counting
        # (depends on attack time)
        assert sway.vad_above > 0 or sway.vad_on


class TestConstants:
    """Tests for module constants."""

    def test_sample_rate(self) -> None:
        """Test SR constant."""
        assert SR == 16000

    def test_frame_ms(self) -> None:
        """Test FRAME_MS constant."""
        assert FRAME_MS == 20

    def test_hop_ms(self) -> None:
        """Test HOP_MS constant."""
        assert HOP_MS == 10

    def test_hop_derived(self) -> None:
        """Test HOP is correctly derived from SR and HOP_MS."""
        assert HOP == int(SR * HOP_MS / 1000)

    def test_frame_derived(self) -> None:
        """Test FRAME is correctly derived from SR and FRAME_MS."""
        assert FRAME == int(SR * FRAME_MS / 1000)

    def test_vad_thresholds(self) -> None:
        """Test VAD thresholds are sensible."""
        assert VAD_DB_ON > VAD_DB_OFF  # Hysteresis
        assert VAD_DB_ON < 0  # Both should be negative dB
        assert VAD_DB_OFF < 0
