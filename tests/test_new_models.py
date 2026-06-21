"""Tests for new FlashAudio architectures and components."""

import torch


class TestHiFiGAN:
    def test_generator_forward(self):
        from flashaudio.models.architectures.hifi_gan import HiFiGANGenerator

        model = HiFiGANGenerator(
            in_channels=80, upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
            upsample_initial_channel=128,
        )
        model.eval()
        mel = torch.randn(1, 80, 50)
        with torch.no_grad():
            audio = model(mel)
        expected_len = 50 * 8 * 8 * 2 * 2
        assert audio.shape[-1] == expected_len

    def test_mpd_forward(self):
        from flashaudio.models.architectures.hifi_gan import MultiPeriodDiscriminator

        mpd = MultiPeriodDiscriminator(periods=(2, 3, 5))
        x = torch.randn(1, 1, 8000)
        outputs, fmaps = mpd(x)
        assert len(outputs) == 3
        assert len(fmaps) == 3

    def test_msd_forward(self):
        from flashaudio.models.architectures.hifi_gan import MultiScaleDiscriminator

        msd = MultiScaleDiscriminator(num_scales=3)
        x = torch.randn(1, 1, 8000)
        outputs, fmaps = msd(x)
        assert len(outputs) == 3


class TestVITS:
    def test_inference(self):
        from flashaudio.models.architectures.vits import VITS

        model = VITS(vocab_size=256, hidden_dim=64, num_layers=2, n_mels=80, num_flows=2)
        model.eval()
        text_ids = torch.randint(0, 256, (1, 10))
        with torch.no_grad():
            out = model.infer(text_ids)
        assert "audio" in out
        assert out["audio"].dim() == 3

    def test_training_forward(self):
        from flashaudio.models.architectures.vits import VITS

        model = VITS(vocab_size=256, hidden_dim=64, num_layers=2, n_mels=80, num_flows=2)
        model.train()
        text_ids = torch.randint(0, 256, (2, 10))
        mel = torch.randn(2, 80, 50)
        out = model(text_ids, mel=mel)
        assert "audio" in out
        assert "kl_loss" in out


class TestWav2Vec2:
    def test_forward(self):
        from flashaudio.models.architectures.wav2vec2 import Wav2Vec2

        model = Wav2Vec2(vocab_size=32, feature_dim=64, hidden_dim=64, num_layers=2, num_heads=4, ff_dim=128)
        model.eval()
        waveform = torch.randn(2, 16000)
        with torch.no_grad():
            out = model(waveform)
        assert "logits" in out
        assert out["logits"].dim() == 3
        assert out["logits"].shape[-1] == 32

    def test_ctc_loss(self):
        from flashaudio.models.architectures.wav2vec2 import Wav2Vec2

        model = Wav2Vec2(vocab_size=32, feature_dim=64, hidden_dim=64, num_layers=2, num_heads=4, ff_dim=128)
        waveform = torch.randn(2, 16000)
        labels = torch.randint(1, 32, (2, 5))
        out = model(waveform, labels=labels)
        assert "loss" in out
        assert out["loss"].dim() == 0

    def test_transcribe(self):
        from flashaudio.models.architectures.wav2vec2 import Wav2Vec2

        model = Wav2Vec2(vocab_size=32, feature_dim=64, hidden_dim=64, num_layers=2, num_heads=4, ff_dim=128)
        waveform = torch.randn(1, 16000)
        decoded = model.transcribe(waveform)
        assert isinstance(decoded, list)
        assert len(decoded) == 1


class TestConformerCTC:
    def test_forward(self):
        from flashaudio.models.architectures.conformer import ConformerCTC

        model = ConformerCTC(vocab_size=32, input_dim=80, encoder_dim=64, num_layers=2, num_heads=4, conv_kernel=15)
        model.eval()
        mel = torch.randn(2, 80, 100)
        with torch.no_grad():
            out = model(mel)
        assert "logits" in out
        assert out["logits"].shape[-1] == 32

    def test_ctc_loss(self):
        from flashaudio.models.architectures.conformer import ConformerCTC

        model = ConformerCTC(vocab_size=32, input_dim=80, encoder_dim=64, num_layers=2, num_heads=4, conv_kernel=15)
        mel = torch.randn(2, 80, 100)
        labels = torch.randint(1, 32, (2, 10))
        out = model(mel, labels=labels)
        assert "loss" in out

    def test_recognize(self):
        from flashaudio.models.architectures.conformer import ConformerCTC

        model = ConformerCTC(vocab_size=32, input_dim=80, encoder_dim=64, num_layers=2, num_heads=4, conv_kernel=15)
        mel = torch.randn(1, 80, 50)
        decoded = model.recognize(mel)
        assert isinstance(decoded, list)


class TestEmotionRecognition:
    def test_forward(self):
        from flashaudio.audio.emotion import EmotionRecognizer

        model = EmotionRecognizer(n_mels=80, hidden_dim=64, num_emotions=7)
        model.eval()
        mel = torch.randn(2, 80, 100)
        with torch.no_grad():
            out = model(mel)
        assert "logits" in out
        assert out["logits"].shape == (2, 7)
        assert "valence_arousal" in out
        assert out["valence_arousal"].shape == (2, 2)

    def test_predict(self):
        from flashaudio.audio.emotion import EmotionRecognizer

        model = EmotionRecognizer(n_mels=80, hidden_dim=64, num_emotions=7)
        mel = torch.randn(1, 80, 100)
        result = model.predict(mel)
        assert "emotion" in result
        assert result["emotion"][0] in ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]


class TestSoundEventDetection:
    def test_forward(self):
        from flashaudio.audio.event_detection import SoundEventDetector

        model = SoundEventDetector(num_classes=10, n_mels=64, base_channels=16)
        model.eval()
        mel = torch.randn(2, 64, 100)
        with torch.no_grad():
            out = model(mel)
        assert "clip_logits" in out
        assert out["clip_logits"].shape == (2, 10)
        assert "frame_logits" in out
        assert "clip_probs" in out

    def test_tag_audio(self):
        from flashaudio.audio.event_detection import SoundEventDetector

        model = SoundEventDetector(num_classes=10, n_mels=64, base_channels=16)
        mel = torch.randn(1, 64, 100)
        result = model.tag_audio(mel, top_k=3)
        assert result["indices"].shape == (1, 3)
        assert result["probabilities"].shape == (1, 3)

    def test_detect_events(self):
        from flashaudio.audio.event_detection import SoundEventDetector

        model = SoundEventDetector(num_classes=10, n_mels=64, base_channels=16)
        mel = torch.randn(1, 64, 100)
        events = model.detect_events(mel, threshold=0.3)
        assert isinstance(events, list)
        assert len(events) == 1


class TestRegistration:
    def test_models_registered(self):
        from flashaudio.registry import MODELS
        assert "HiFiGAN" in MODELS
        assert "VITS" in MODELS
        assert "Wav2Vec2" in MODELS
        assert "ConformerCTC" in MODELS

    def test_emotion_registered(self):
        from flashaudio.registry import MODELS
        assert "EmotionRecognition" in MODELS

    def test_event_detection_registered(self):
        from flashaudio.registry import MODELS
        assert "SoundEventDetector" in MODELS
