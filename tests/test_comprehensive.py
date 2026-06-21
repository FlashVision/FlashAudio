"""Comprehensive test suite for FlashAudio covering all architectures,
pipelines, features, metrics, diarization, and CLI."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from flashaudio.registry import MODELS, PIPELINES


# ===================================================================
# Architecture: HiFi-GAN
# ===================================================================


class TestHiFiGAN:
    def test_generator_forward(self):
        from flashaudio.models.architectures.hifi_gan import HiFiGANGenerator

        gen = HiFiGANGenerator(
            in_channels=80, upsample_rates=(4, 4), upsample_kernel_sizes=(8, 8), upsample_initial_channel=64
        )
        gen.eval()
        mel = torch.randn(1, 80, 10)
        with torch.no_grad():
            wav = gen(mel)
        assert wav.dim() == 3
        assert wav.shape[1] == 1

    def test_generator_infer(self):
        from flashaudio.models.architectures.hifi_gan import HiFiGANGenerator

        gen = HiFiGANGenerator(
            in_channels=80, upsample_rates=(4,), upsample_kernel_sizes=(8,), upsample_initial_channel=64
        )
        mel = torch.randn(1, 80, 10)
        wav = gen.infer(mel)
        assert wav.dim() == 3

    def test_resblock1(self):
        from flashaudio.models.architectures.hifi_gan import ResBlock1

        block = ResBlock1(64, kernel_size=3, dilations=(1, 3))
        x = torch.randn(1, 64, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_resblock2(self):
        from flashaudio.models.architectures.hifi_gan import ResBlock2

        block = ResBlock2(64, kernel_size=3, dilations=(1, 3))
        x = torch.randn(1, 64, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_multi_period_discriminator(self):
        from flashaudio.models.architectures.hifi_gan import MultiPeriodDiscriminator

        mpd = MultiPeriodDiscriminator(periods=(2, 3))
        x = torch.randn(1, 1, 256)
        outputs, fmaps = mpd(x)
        assert len(outputs) == 2
        assert len(fmaps) == 2

    def test_multi_scale_discriminator(self):
        from flashaudio.models.architectures.hifi_gan import MultiScaleDiscriminator

        msd = MultiScaleDiscriminator(num_scales=2)
        x = torch.randn(1, 1, 256)
        outputs, fmaps = msd(x)
        assert len(outputs) == 2

    def test_losses(self):
        from flashaudio.models.architectures.hifi_gan import HiFiGANLoss

        disc_real = [torch.ones(1, 10)]
        disc_gen = [torch.zeros(1, 10)]
        g_loss = HiFiGANLoss.generator_loss(disc_gen)
        d_loss = HiFiGANLoss.discriminator_loss(disc_real, disc_gen)
        assert g_loss.dim() == 0
        assert d_loss.dim() == 0


# ===================================================================
# Architecture: VITS
# ===================================================================


class TestVITS:
    def test_text_encoder(self):
        from flashaudio.models.architectures.vits import TextEncoder

        enc = TextEncoder(vocab_size=256, hidden_dim=32, num_layers=1, num_heads=2, filter_size=64)
        text_ids = torch.randint(0, 256, (2, 10))
        out = enc(text_ids)
        assert "mu" in out
        assert "log_sigma" in out
        assert out["mu"].shape == (2, 10, 32)

    def test_posterior_encoder(self):
        from flashaudio.models.architectures.vits import PosteriorEncoder

        enc = PosteriorEncoder(in_channels=80, hidden_dim=32, num_layers=2)
        mel = torch.randn(2, 80, 20)
        out = enc(mel)
        assert "z" in out
        assert "mu" in out

    def test_flow_module(self):
        from flashaudio.models.architectures.vits import FlowModule

        flow = FlowModule(channels=32, num_flows=2, hidden_dim=32)
        x = torch.randn(1, 32, 10)
        y, log_det = flow(x, reverse=False)
        assert y.shape == x.shape

    def test_duration_predictor(self):
        from flashaudio.models.architectures.vits import DurationPredictor

        dp = DurationPredictor(in_channels=32, hidden_dim=32)
        x = torch.randn(1, 32, 10)
        dur = dp(x)
        assert dur.shape == (1, 10)

    def test_vits_training_forward(self):
        from flashaudio.models.architectures.vits import VITS

        model = VITS(vocab_size=64, hidden_dim=32, num_layers=1, n_mels=16, num_flows=1)
        text_ids = torch.randint(0, 64, (1, 8))
        mel = torch.randn(1, 16, 20)
        out = model(text_ids, mel=mel)
        assert "audio" in out
        assert "kl_loss" in out

    def test_vits_infer(self):
        from flashaudio.models.architectures.vits import VITS

        model = VITS(vocab_size=64, hidden_dim=32, num_layers=1, n_mels=16, num_flows=1)
        model.eval()
        text_ids = torch.randint(0, 64, (1, 5))
        out = model.infer(text_ids)
        assert "audio" in out

    def test_text_to_ids(self):
        from flashaudio.models.architectures.vits import VITS

        model = VITS(vocab_size=256, hidden_dim=32, num_layers=1, n_mels=16, num_flows=1)
        ids = model.text_to_ids("hello")
        assert ids.shape == (1, 5)

    def test_monotonic_alignment_search(self):
        from flashaudio.models.architectures.vits import monotonic_alignment_search

        log_p = torch.randn(4, 8)
        durs = monotonic_alignment_search(log_p, text_len=4, mel_len=8)
        assert durs.sum().item() == 8
        assert durs.shape == (4,)


# ===================================================================
# Architecture: Wav2Vec2
# ===================================================================


class TestWav2Vec2:
    def test_forward(self):
        from flashaudio.models.architectures.wav2vec2 import Wav2Vec2

        model = Wav2Vec2(vocab_size=32, feature_dim=64, hidden_dim=64, num_layers=1, num_heads=4, ff_dim=128)
        model.eval()
        waveform = torch.randn(1, 3200)
        with torch.no_grad():
            out = model(waveform)
        assert "logits" in out
        assert "features" in out

    def test_ctc_loss(self):
        from flashaudio.models.architectures.wav2vec2 import Wav2Vec2

        model = Wav2Vec2(vocab_size=32, feature_dim=64, hidden_dim=64, num_layers=1, num_heads=4, ff_dim=128)
        waveform = torch.randn(1, 3200)
        labels = torch.randint(1, 32, (1, 5))
        out = model(waveform, labels=labels)
        assert "loss" in out
        assert out["loss"].dim() == 0

    def test_greedy_decode(self):
        from flashaudio.models.architectures.wav2vec2 import CTCDecoder

        dec = CTCDecoder(in_dim=32, vocab_size=10)
        logits = torch.randn(1, 20, 10)
        decoded = dec.decode_greedy(logits)
        assert isinstance(decoded, list)
        assert isinstance(decoded[0], list)

    def test_extract_features(self):
        from flashaudio.models.architectures.wav2vec2 import Wav2Vec2

        model = Wav2Vec2(vocab_size=32, feature_dim=64, hidden_dim=64, num_layers=1, num_heads=4, ff_dim=128)
        model.eval()
        waveform = torch.randn(1, 3200)
        with torch.no_grad():
            feats = model.extract_features(waveform)
        assert feats.dim() == 3


# ===================================================================
# Architecture: Conformer
# ===================================================================


class TestConformer:
    def test_conformer_ctc_forward(self):
        from flashaudio.models.architectures.conformer import ConformerCTC

        model = ConformerCTC(
            vocab_size=32, input_dim=16, encoder_dim=32, num_layers=1, num_heads=4, conv_kernel=3, subsampling=2
        )
        model.eval()
        features = torch.randn(1, 16, 40)
        with torch.no_grad():
            out = model(features)
        assert "logits" in out

    def test_conformer_with_ctc_loss(self):
        from flashaudio.models.architectures.conformer import ConformerCTC

        model = ConformerCTC(
            vocab_size=32, input_dim=16, encoder_dim=32, num_layers=1, num_heads=4, conv_kernel=3, subsampling=2
        )
        features = torch.randn(1, 16, 40)
        labels = torch.randint(1, 32, (1, 5))
        out = model(features, labels=labels)
        assert "loss" in out

    def test_greedy_decode(self):
        from flashaudio.models.architectures.conformer import ConformerCTC

        model = ConformerCTC(
            vocab_size=32, input_dim=16, encoder_dim=32, num_layers=1, num_heads=4, conv_kernel=3, subsampling=2
        )
        logits = torch.randn(1, 10, 32)
        decoded = model.decode_greedy(logits)
        assert isinstance(decoded, list)

    def test_conformer_block(self):
        from flashaudio.models.architectures.conformer import ConformerBlock

        block = ConformerBlock(dim=32, num_heads=4, conv_kernel=3)
        x = torch.randn(2, 10, 32)
        out = block(x)
        assert out.shape == x.shape


# ===================================================================
# Voice Encoder / Speaker Embedding
# ===================================================================


class TestVoiceEncoder:
    def test_forward_mel(self):
        from flashaudio.models.architectures.voice_encoder import VoiceEncoderModel

        model = VoiceEncoderModel(input_dim=80, channels=64, embedding_dim=32, num_blocks=1)
        model.eval()
        mel = torch.randn(2, 80, 100)
        with torch.no_grad():
            emb = model(mel)
        assert emb.shape == (2, 32)
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_se_block(self):
        from flashaudio.models.architectures.voice_encoder import SEBlock

        se = SEBlock(64, reduction=4)
        x = torch.randn(1, 64, 20)
        out = se(x)
        assert out.shape == x.shape

    def test_attentive_pooling(self):
        from flashaudio.models.architectures.voice_encoder import AttentiveStatisticsPooling

        asp = AttentiveStatisticsPooling(64, attention_dim=16)
        x = torch.randn(2, 64, 20)
        out = asp(x)
        assert out.shape == (2, 128)

    def test_extract_embedding(self):
        from flashaudio.models.architectures.voice_encoder import VoiceEncoderModel

        model = VoiceEncoderModel(input_dim=80, channels=64, embedding_dim=32, num_blocks=1)
        model.eval()
        mel = torch.randn(80, 100)
        emb = model.extract_embedding(mel)
        assert emb.shape == (32,)


# ===================================================================
# Emotion Recognition
# ===================================================================


class TestEmotionRecognition:
    def test_forward(self):
        from flashaudio.audio.emotion import EmotionRecognizer

        model = EmotionRecognizer(n_mels=16, hidden_dim=32, num_emotions=7, prosody_dim=16)
        model.eval()
        mel = torch.randn(2, 16, 50)
        with torch.no_grad():
            out = model(mel)
        assert "logits" in out
        assert out["logits"].shape == (2, 7)
        assert "probabilities" in out
        assert "valence_arousal" in out
        assert out["valence_arousal"].shape == (2, 2)

    def test_predict(self):
        from flashaudio.audio.emotion import EmotionRecognizer

        model = EmotionRecognizer(n_mels=16, hidden_dim=32, num_emotions=7, prosody_dim=16)
        mel = torch.randn(2, 16, 50)
        preds = model.predict(mel)
        assert "emotion" in preds
        assert len(preds["emotion"]) == 2

    def test_compute_loss(self):
        from flashaudio.audio.emotion import EmotionRecognizer

        model = EmotionRecognizer(n_mels=16, hidden_dim=32, num_emotions=7, prosody_dim=16)
        mel = torch.randn(2, 16, 50)
        out = model(mel)
        targets = torch.randint(0, 7, (2,))
        losses = model.compute_loss(out, targets)
        assert "classification" in losses
        assert "total" in losses


# ===================================================================
# Sound Event Detection
# ===================================================================


class TestSoundEventDetection:
    def test_forward(self):
        from flashaudio.audio.event_detection import SoundEventDetector

        model = SoundEventDetector(num_classes=10, n_mels=64, base_channels=8)
        model.eval()
        mel = torch.randn(1, 64, 100)
        with torch.no_grad():
            out = model(mel)
        assert "clip_logits" in out
        assert out["clip_logits"].shape[1] == 10
        assert "clip_probs" in out

    def test_detect_events(self):
        from flashaudio.audio.event_detection import SoundEventDetector

        model = SoundEventDetector(num_classes=10, n_mels=64, base_channels=8)
        mel = torch.randn(1, 64, 100)
        events = model.detect_events(mel, threshold=0.5)
        assert isinstance(events, list)
        assert len(events) == 1

    def test_tag_audio(self):
        from flashaudio.audio.event_detection import SoundEventDetector

        model = SoundEventDetector(num_classes=10, n_mels=64, base_channels=8)
        mel = torch.randn(1, 64, 100)
        tags = model.tag_audio(mel, top_k=3)
        assert "indices" in tags
        assert tags["indices"].shape == (1, 3)

    def test_compute_loss(self):
        from flashaudio.audio.event_detection import SoundEventDetector

        model = SoundEventDetector(num_classes=10, n_mels=64, base_channels=8)
        mel = torch.randn(1, 64, 100)
        out = model(mel)
        targets = torch.zeros(1, 10)
        targets[0, 3] = 1.0
        losses = model.compute_loss(out, targets)
        assert "clip_loss" in losses


# ===================================================================
# Audio Features (mel spectrogram, MFCC)
# ===================================================================


class TestAudioFeatures:
    def test_mel_spectrogram(self):
        from flashaudio.audio.features import AudioFeatureExtractor

        extractor = AudioFeatureExtractor(sample_rate=16000, n_fft=256, hop_length=128, n_mels=32)
        waveform = torch.randn(1, 8000)
        mel = extractor.mel_spectrogram(waveform)
        assert mel.dim() == 3
        assert mel.shape[1] == 32

    def test_mfcc(self):
        from flashaudio.audio.features import AudioFeatureExtractor

        extractor = AudioFeatureExtractor(sample_rate=16000, n_fft=256, hop_length=128, n_mels=32, n_mfcc=13)
        waveform = torch.randn(1, 8000)
        mfcc = extractor.mfcc(waveform)
        assert mfcc.dim() == 3
        assert mfcc.shape[1] == 13

    def test_mel_1d_input(self):
        from flashaudio.audio.features import AudioFeatureExtractor

        extractor = AudioFeatureExtractor(sample_rate=16000, n_fft=256, hop_length=128, n_mels=32)
        waveform = torch.randn(8000)
        mel = extractor.mel_spectrogram(waveform)
        assert mel.dim() == 3


# ===================================================================
# Metrics: WER, CER
# ===================================================================


class TestAudioMetrics:
    def test_wer_perfect(self):
        from flashaudio.analytics.metrics import compute_wer

        wer = compute_wer("hello world", "hello world")
        assert wer == 0.0

    def test_wer_one_error(self):
        from flashaudio.analytics.metrics import compute_wer

        wer = compute_wer("hello world", "hello earth")
        assert 0.0 < wer <= 1.0

    def test_cer_perfect(self):
        from flashaudio.analytics.metrics import compute_cer

        cer = compute_cer("hello", "hello")
        assert cer == 0.0

    def test_cer_batch(self):
        from flashaudio.analytics.metrics import compute_cer

        cer = compute_cer(["hello", "world"], ["helo", "world"])
        assert 0.0 <= cer <= 1.0

    def test_mos(self):
        from flashaudio.analytics.metrics import compute_mos

        waveform = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
        mos = compute_mos(waveform, sample_rate=16000)
        assert 1.0 <= mos <= 5.0

    def test_pesq(self):
        from flashaudio.analytics.metrics import compute_pesq

        ref = np.sin(np.linspace(0, 100, 4000)).astype(np.float32)
        deg = ref + np.random.randn(4000).astype(np.float32) * 0.01
        pesq = compute_pesq(ref, deg, sample_rate=16000)
        assert 1.0 <= pesq <= 4.5


# ===================================================================
# Diarization
# ===================================================================


class TestDiarization:
    def test_segment_audio(self):
        from flashaudio.speech.diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device="cpu", segment_duration=0.5, hop_duration=0.25, num_speakers=2)
        waveform = torch.randn(16000)
        segments = diarizer._segment_audio(waveform, sample_rate=16000)
        assert len(segments) > 0
        for start, end, chunk in segments:
            assert end > start

    def test_merge_adjacent(self):
        from flashaudio.speech.diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device="cpu")
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01"},
        ]
        merged = diarizer._merge_adjacent(segments)
        assert len(merged) == 2
        assert merged[0]["end"] == 2.0

    def test_cluster_embeddings(self):
        from flashaudio.speech.diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device="cpu", num_speakers=2, similarity_threshold=0.3)
        embeddings = torch.randn(6, 32)
        embeddings[:3] += 5.0
        labels = diarizer._cluster_embeddings(embeddings)
        assert len(labels) == 6


# ===================================================================
# CLI
# ===================================================================


class TestAudioCLI:
    def test_main_no_command(self):
        from flashaudio.cli import main

        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["flashaudio"]):
                main()
        assert exc.value.code == 0


# ===================================================================
# Registry
# ===================================================================


class TestAudioRegistries:
    def test_models_registered(self):
        assert "HiFiGAN" in MODELS
        assert "VITS" in MODELS
        assert "Wav2Vec2" in MODELS
        assert "ConformerCTC" in MODELS
        assert "VoiceEncoder" in MODELS
        assert "EmotionRecognition" in MODELS
        assert "SoundEventDetector" in MODELS

    def test_pipelines_registered(self):
        assert "TextToSpeech" in PIPELINES
        assert "SpeechToText" in PIPELINES
        assert "SpeakerDiarizer" in PIPELINES


# ===================================================================
# TTS / STT pipeline classes
# ===================================================================


class TestTTSPipeline:
    def test_text_to_ids(self):
        from flashaudio.speech.tts import TextToSpeech

        tts = TextToSpeech(device="cpu")
        ids = tts._text_to_ids("hello")
        assert ids.shape == (1, 5)

    def test_create_segments(self):
        from flashaudio.speech.stt import SpeechToText

        stt = SpeechToText.__new__(SpeechToText)
        segments = stt._create_segments("Hello world. How are you?", 5.0)
        assert len(segments) == 2
        assert segments[0]["text"] == "Hello world."


class TestSTTPipeline:
    def test_stt_init(self):
        from flashaudio.speech.stt import SpeechToText

        stt = SpeechToText(device="cpu")
        assert stt.model_id == "openai/whisper-base"

    def test_supported_models(self):
        from flashaudio.speech.stt import SpeechToText

        assert len(SpeechToText.SUPPORTED_MODELS) >= 5
