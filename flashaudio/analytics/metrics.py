"""Audio and speech quality metrics.

Implements WER, CER, MOS, and PESQ computations for evaluating
speech recognition and synthesis quality.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np


def compute_wer(
    references: Union[str, List[str]],
    hypotheses: Union[str, List[str]],
) -> float:
    """Compute Word Error Rate (WER).

    WER = (S + D + I) / N where S=substitutions, D=deletions,
    I=insertions, N=number of words in reference.

    Args:
        references: Reference transcript(s).
        hypotheses: Hypothesis transcript(s).

    Returns:
        WER as a float (0.0 = perfect, >1.0 possible with insertions).
    """
    if isinstance(references, str):
        references = [references]
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]

    total_errors = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.strip().lower().split()
        hyp_words = hyp.strip().lower().split()

        errors = _levenshtein_distance(ref_words, hyp_words)
        total_errors += errors
        total_words += len(ref_words)

    return total_errors / max(total_words, 1)


def compute_cer(
    references: Union[str, List[str]],
    hypotheses: Union[str, List[str]],
) -> float:
    """Compute Character Error Rate (CER).

    Same as WER but operates at the character level.

    Args:
        references: Reference transcript(s).
        hypotheses: Hypothesis transcript(s).

    Returns:
        CER as a float.
    """
    if isinstance(references, str):
        references = [references]
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]

    total_errors = 0
    total_chars = 0

    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref.strip().lower())
        hyp_chars = list(hyp.strip().lower())

        errors = _levenshtein_distance(ref_chars, hyp_chars)
        total_errors += errors
        total_chars += len(ref_chars)

    return total_errors / max(total_chars, 1)


def compute_mos(
    waveform: np.ndarray,
    sample_rate: int = 16000,
) -> float:
    """Estimate Mean Opinion Score (MOS) for synthesized speech.

    Uses a signal-based proxy: combination of SNR, spectral flatness,
    and harmonic-to-noise ratio. Not a perceptual model — use for
    relative comparisons only.

    Real MOS requires human listeners rating on a 1-5 scale.

    Args:
        waveform: Audio signal as numpy array.
        sample_rate: Sample rate.

    Returns:
        Estimated MOS score (1.0 - 5.0).
    """
    if isinstance(waveform, (list, tuple)):
        waveform = np.array(waveform)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)

    waveform = waveform.astype(np.float64)
    if waveform.max() > 1.0:
        waveform = waveform / 32768.0

    rms = np.sqrt(np.mean(waveform ** 2))
    if rms < 1e-10:
        return 1.0

    signal_power = np.mean(waveform ** 2)
    noise_est = np.mean(np.diff(waveform) ** 2) / 2
    snr = 10 * np.log10(signal_power / max(noise_est, 1e-10))
    snr = np.clip(snr, 0, 50)

    fft_mag = np.abs(np.fft.rfft(waveform))
    fft_mag = fft_mag[fft_mag > 0]
    if len(fft_mag) > 0:
        spectral_flatness = np.exp(np.mean(np.log(fft_mag + 1e-10))) / (np.mean(fft_mag) + 1e-10)
    else:
        spectral_flatness = 0.0

    zcr = np.mean(np.abs(np.diff(np.sign(waveform)))) / 2
    voicing_score = 1.0 - min(zcr * 5, 1.0)

    snr_score = np.clip(snr / 40 * 4 + 1, 1, 5)
    flatness_score = np.clip((1 - spectral_flatness) * 4 + 1, 1, 5)
    voicing_score = np.clip(voicing_score * 4 + 1, 1, 5)

    mos = 0.5 * snr_score + 0.3 * flatness_score + 0.2 * voicing_score
    return float(np.clip(mos, 1.0, 5.0))


def compute_pesq(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
) -> float:
    """Compute PESQ-like perceptual quality score.

    Simplified proxy based on spectral distortion and SNR.
    For true ITU-T P.862 PESQ, use the pesq Python package.

    Args:
        reference: Reference audio signal.
        degraded: Degraded/synthesized audio signal.
        sample_rate: Sample rate (must match for both).

    Returns:
        Quality score (approximately 1.0 - 4.5 range).
    """
    if isinstance(reference, (list, tuple)):
        reference = np.array(reference)
    if isinstance(degraded, (list, tuple)):
        degraded = np.array(degraded)

    if reference.ndim == 2:
        reference = reference.mean(axis=0)
    if degraded.ndim == 2:
        degraded = degraded.mean(axis=0)

    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len].astype(np.float64)
    degraded = degraded[:min_len].astype(np.float64)

    if np.max(np.abs(reference)) > 1.0:
        reference = reference / 32768.0
    if np.max(np.abs(degraded)) > 1.0:
        degraded = degraded / 32768.0

    error = reference - degraded
    signal_power = np.mean(reference ** 2)
    error_power = np.mean(error ** 2)

    if error_power < 1e-10:
        seg_snr = 45.0
    else:
        seg_snr = 10 * np.log10(signal_power / error_power)
    seg_snr = np.clip(seg_snr, -10, 45)

    n_fft = 512
    hop = 256
    ref_frames = _stft_magnitude(reference, n_fft, hop)
    deg_frames = _stft_magnitude(degraded, n_fft, hop)

    min_frames = min(ref_frames.shape[0], deg_frames.shape[0])
    ref_frames = ref_frames[:min_frames]
    deg_frames = deg_frames[:min_frames]

    spectral_dist = np.mean(np.sqrt(np.mean((ref_frames - deg_frames) ** 2, axis=1)))

    snr_score = np.clip((seg_snr + 10) / 55 * 3.5 + 1, 1, 4.5)
    dist_penalty = np.clip(spectral_dist * 2, 0, 2)

    pesq_score = snr_score - dist_penalty
    return float(np.clip(pesq_score, 1.0, 4.5))


def _levenshtein_distance(seq1: list, seq2: list) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(seq1), len(seq2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[m][n]


def _stft_magnitude(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Compute STFT magnitude frames."""
    window = np.hanning(n_fft)
    num_frames = (len(signal) - n_fft) // hop + 1

    if num_frames <= 0:
        return np.zeros((1, n_fft // 2 + 1))

    frames = np.zeros((num_frames, n_fft // 2 + 1))
    for i in range(num_frames):
        start = i * hop
        frame = signal[start:start + n_fft] * window
        fft = np.fft.rfft(frame)
        frames[i] = np.abs(fft)

    return frames
