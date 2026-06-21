"""Unit tests for audio model architectures."""

import pytest
import torch


class TestAudioClassifier:
    def test_forward_from_waveform(self):
        from flashaudio.models.architectures.audio_classifier import AudioClassifierModel

        model = AudioClassifierModel(num_classes=10)
        model.eval()

        waveform = torch.randn(2, 16000 * 3)
        with torch.no_grad():
            output = model(waveform)

        assert output.shape == (2, 10)

    def test_forward_from_mel(self):
        from flashaudio.models.architectures.audio_classifier import AudioClassifierModel

        model = AudioClassifierModel(num_classes=10, n_mels=80)
        model.eval()

        mel = torch.randn(2, 80, 100)
        with torch.no_grad():
            output = model(mel)

        assert output.shape == (2, 10)

    def test_predict(self):
        from flashaudio.models.architectures.audio_classifier import AudioClassifierModel

        model = AudioClassifierModel(num_classes=20)
        waveform = torch.randn(1, 16000 * 2)

        result = model.predict(waveform, top_k=5)
        assert result["probabilities"].shape == (1, 5)
        assert result["indices"].shape == (1, 5)


class TestTTSModel:
    def test_forward(self):
        from flashaudio.models.architectures.tts_model import TTSModel

        model = TTSModel(vocab_size=256, n_mels=80)
        model.eval()

        text_ids = torch.randint(0, 256, (1, 20))
        with torch.no_grad():
            output = model(text_ids, max_steps=10)

        assert "mel" in output
        assert "mel_postnet" in output
        assert "stop_logits" in output
        assert output["mel"].shape[1] == 80
        assert output["mel"].shape[2] == 10

    def test_text_to_ids(self):
        from flashaudio.models.architectures.tts_model import TTSModel

        model = TTSModel()
        ids = model.text_to_ids("hello")
        assert ids.shape == (1, 5)
        assert ids.dtype == torch.long


class TestVoiceEncoder:
    def test_forward_from_mel(self):
        from flashaudio.models.architectures.voice_encoder import VoiceEncoderModel

        model = VoiceEncoderModel(input_dim=80, embedding_dim=192)
        model.eval()

        mel = torch.randn(2, 80, 200)
        with torch.no_grad():
            embeddings = model(mel)

        assert embeddings.shape == (2, 192)
        norms = embeddings.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_extract_embedding(self):
        from flashaudio.models.architectures.voice_encoder import VoiceEncoderModel

        model = VoiceEncoderModel(embedding_dim=128)
        model.eval()

        mel = torch.randn(80, 200)
        emb = model.extract_embedding(mel)
        assert emb.shape == (128,)


class TestLoRA:
    def test_apply_and_merge(self):
        from flashaudio.models.lora import apply_lora, merge_lora_weights, LoRALinear

        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

        original_params = sum(p.numel() for p in model.parameters())
        model = apply_lora(model, rank=4, alpha=8, target_modules={"0", "2"})

        has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
        assert has_lora

        model = merge_lora_weights(model)
        has_lora_after = any(isinstance(m, LoRALinear) for m in model.modules())
        assert not has_lora_after
