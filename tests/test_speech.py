"""Unit tests for speech and audio pipelines."""

import pytest
import numpy as np
import torch


class TestMetrics:
    def test_wer_perfect(self):
        from flashaudio.analytics.metrics import compute_wer
        assert compute_wer("hello world", "hello world") == 0.0

    def test_wer_all_wrong(self):
        from flashaudio.analytics.metrics import compute_wer
        wer = compute_wer("hello world", "goodbye earth")
        assert wer == 1.0

    def test_wer_partial(self):
        from flashaudio.analytics.metrics import compute_wer
        wer = compute_wer("the quick brown fox", "the quik brown fox")
        assert 0 < wer < 1

    def test_cer(self):
        from flashaudio.analytics.metrics import compute_cer
        cer = compute_cer("hello", "hallo")
        assert 0 < cer < 1

    def test_cer_perfect(self):
        from flashaudio.analytics.metrics import compute_cer
        assert compute_cer("test", "test") == 0.0

    def test_mos_valid_range(self):
        from flashaudio.analytics.metrics import compute_mos
        signal = np.random.randn(16000).astype(np.float64) * 0.1
        mos = compute_mos(signal)
        assert 1.0 <= mos <= 5.0

    def test_mos_silence(self):
        from flashaudio.analytics.metrics import compute_mos
        silence = np.zeros(16000)
        mos = compute_mos(silence)
        assert mos == 1.0

    def test_pesq_valid_range(self):
        from flashaudio.analytics.metrics import compute_pesq
        ref = np.random.randn(16000).astype(np.float64)
        deg = ref + np.random.randn(16000) * 0.01
        score = compute_pesq(ref, deg)
        assert 1.0 <= score <= 4.5

    def test_wer_batch(self):
        from flashaudio.analytics.metrics import compute_wer
        refs = ["hello world", "foo bar baz"]
        hyps = ["hello world", "foo baz bar"]
        wer = compute_wer(refs, hyps)
        assert 0 <= wer <= 1


class TestAudioFeatures:
    def test_mel_spectrogram(self):
        from flashaudio.audio.features import AudioFeatureExtractor

        extractor = AudioFeatureExtractor(sample_rate=16000, n_mels=80)
        waveform = torch.randn(1, 16000)
        mel = extractor.mel_spectrogram(waveform)

        assert mel.shape[1] == 80
        assert mel.dim() == 3

    def test_mfcc(self):
        from flashaudio.audio.features import AudioFeatureExtractor

        extractor = AudioFeatureExtractor(n_mfcc=13)
        waveform = torch.randn(16000)
        mfcc = extractor.mfcc(waveform)

        assert mfcc.shape[1] == 13

    def test_extract_all(self):
        from flashaudio.audio.features import AudioFeatureExtractor

        extractor = AudioFeatureExtractor()
        waveform = torch.randn(16000 * 2)
        features = extractor.extract_all(waveform)

        assert "mel_spectrogram" in features
        assert "mfcc" in features
        assert "chromagram" in features
        assert "spectral_centroid" in features
        assert "zero_crossing_rate" in features
        assert "rms_energy" in features


class TestAudioUtils:
    def test_pad_or_trim_pad(self):
        from flashaudio.data.audio_utils import pad_or_trim

        waveform = torch.randn(8000)
        result = pad_or_trim(waveform, 16000)
        assert result.shape[0] == 16000

    def test_pad_or_trim_trim(self):
        from flashaudio.data.audio_utils import pad_or_trim

        waveform = torch.randn(32000)
        result = pad_or_trim(waveform, 16000)
        assert result.shape[0] == 16000

    def test_pad_or_trim_exact(self):
        from flashaudio.data.audio_utils import pad_or_trim

        waveform = torch.randn(16000)
        result = pad_or_trim(waveform, 16000)
        assert torch.equal(result, waveform)

    def test_compute_mel_spectrogram(self):
        from flashaudio.data.audio_utils import compute_mel_spectrogram

        waveform = torch.randn(1, 16000)
        mel = compute_mel_spectrogram(waveform, n_mels=64)
        assert mel.shape[1] == 64


class TestConfig:
    def test_default_config(self):
        from flashaudio.cfg.config import FlashAudioConfig

        cfg = FlashAudioConfig()
        assert cfg.model.model_id == "openai/whisper-base"
        assert cfg.train.epochs == 10
        assert cfg.audio.sample_rate == 16000

    def test_get_config_with_overrides(self):
        from flashaudio.cfg.config import get_config

        cfg = get_config(model__model_id="openai/whisper-small")
        assert cfg.model.model_id == "openai/whisper-base"

        cfg = get_config(**{"model.model_id": "openai/whisper-small"})
        assert cfg.model.model_id == "openai/whisper-small"

    def test_to_dict(self):
        from flashaudio.cfg.config import FlashAudioConfig

        cfg = FlashAudioConfig()
        d = cfg.to_dict()
        assert "model.model_id" in d
        assert "train.epochs" in d
