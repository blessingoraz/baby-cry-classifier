"""
Unit tests for src/preprocess.py audio preprocessing utilities.

Run with: pytest tests/test_preprocess.py -v
"""
import pytest
import numpy as np
import torch
import tempfile
import soundfile as sf
from pathlib import Path

from src.preprocess import (
    load_and_fix_length,
    audio_to_mel_tensor,
    TARGET_SR,
    FIXED_LEN,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
)


class TestLoadAndFixLength:
    """Test load_and_fix_length function."""

    @pytest.fixture
    def short_audio_file(self):
        """Create a temporary short audio file (shorter than FIXED_LEN)."""
        sr = TARGET_SR
        duration = 2.0  # 2 seconds, should be padded
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            yield f.name
        
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def long_audio_file(self):
        """Create a temporary long audio file (longer than FIXED_LEN)."""
        sr = TARGET_SR
        duration = 10.0  # 10 seconds, should be trimmed
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            yield f.name
        
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def exact_length_audio_file(self):
        """Create an audio file with exact FIXED_LEN."""
        sr = TARGET_SR
        duration = CLIP_SECONDS = 7.0  # Exact duration
        t = np.linspace(0, duration, FIXED_LEN)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            yield f.name
        
        Path(f.name).unlink(missing_ok=True)

    def test_short_audio_is_padded(self, short_audio_file):
        """Test that short audio is padded to FIXED_LEN."""
        y = load_and_fix_length(short_audio_file)
        
        assert len(y) == FIXED_LEN
        # Check that padding is zeros
        assert y[-1] == 0.0

    def test_long_audio_is_trimmed(self, long_audio_file):
        """Test that long audio is trimmed to FIXED_LEN."""
        y = load_and_fix_length(long_audio_file)
        
        assert len(y) == FIXED_LEN

    def test_exact_length_audio_unchanged(self, exact_length_audio_file):
        """Test that audio with exact length is unchanged."""
        y = load_and_fix_length(exact_length_audio_file)
        
        assert len(y) == FIXED_LEN

    def test_return_type_is_numpy_array(self, short_audio_file):
        """Test that return type is numpy array."""
        y = load_and_fix_length(short_audio_file)
        
        assert isinstance(y, np.ndarray)
        assert y.dtype in [np.float32, np.float64]

    def test_mono_conversion(self, short_audio_file):
        """Test that audio is loaded as mono (1D)."""
        y = load_and_fix_length(short_audio_file)
        
        assert y.ndim == 1

    def test_custom_target_sr(self):
        """Test with custom target sampling rate."""
        custom_sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(custom_sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, custom_sr)
            
            y = load_and_fix_length(f.name, target_sr=custom_sr)
            
            assert len(y) == FIXED_LEN
            
            Path(f.name).unlink(missing_ok=True)

    def test_custom_fixed_len(self, short_audio_file):
        """Test with custom fixed length."""
        custom_fixed_len = 10000
        y = load_and_fix_length(short_audio_file, fixed_len=custom_fixed_len)
        
        assert len(y) == custom_fixed_len

    def test_file_not_found_raises_error(self):
        """Test that nonexistent file raises error."""
        with pytest.raises(Exception):
            load_and_fix_length("nonexistent_file.wav")


class TestAudioToMelTensor:
    """Test audio_to_mel_tensor function."""

    @pytest.fixture
    def sample_audio_file(self):
        """Create a sample audio file."""
        sr = TARGET_SR
        duration = 7.0
        t = np.linspace(0, duration, int(sr * duration))
        # Mix of frequencies for interesting spectrogram
        audio = (np.sin(2 * np.pi * 440 * t) + 
                 0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            yield f.name
        
        Path(f.name).unlink(missing_ok=True)

    def test_return_type_is_torch_tensor(self, sample_audio_file):
        """Test that return type is torch tensor."""
        x = audio_to_mel_tensor(sample_audio_file)
        
        assert isinstance(x, torch.Tensor)

    def test_output_shape(self, sample_audio_file):
        """Test that output has correct shape."""
        x = audio_to_mel_tensor(sample_audio_file)
        
        # Expected shape: (batch=1, channels=1, height=224, width=224)
        assert x.shape == (1, 1, 224, 224)

    def test_output_dtype(self, sample_audio_file):
        """Test that output tensor has float32 dtype."""
        x = audio_to_mel_tensor(sample_audio_file)
        
        assert x.dtype == torch.float32

    def test_output_values_normalized(self, sample_audio_file):
        """Test that output values are normalized (roughly 0-1)."""
        x = audio_to_mel_tensor(sample_audio_file)
        
        # Values should be normalized to 0-1 range
        assert x.min().item() >= 0.0
        assert x.max().item() <= 1.0

    def test_output_requires_no_grad(self, sample_audio_file):
        """Test that output tensor does not require gradients."""
        x = audio_to_mel_tensor(sample_audio_file)
        
        assert not x.requires_grad

    def test_different_audio_produces_different_tensor(self):
        """Test that different audio files produce different tensors."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1:
            # First audio: 440 Hz sine wave
            sr = TARGET_SR
            duration = 7.0
            t = np.linspace(0, duration, int(sr * duration))
            audio1 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            sf.write(f1.name, audio1, sr)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
                # Second audio: 880 Hz sine wave
                audio2 = np.sin(2 * np.pi * 880 * t).astype(np.float32)
                sf.write(f2.name, audio2, sr)
                
                x1 = audio_to_mel_tensor(f1.name)
                x2 = audio_to_mel_tensor(f2.name)
                
                # Tensors should be different
                assert not torch.allclose(x1, x2)
                
                Path(f2.name).unlink(missing_ok=True)
        
        Path(f1.name).unlink(missing_ok=True)

    def test_silent_audio(self):
        """Test with silent audio (all zeros)."""
        sr = TARGET_SR
        duration = 7.0
        audio = np.zeros(int(sr * duration), dtype=np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            
            x = audio_to_mel_tensor(f.name)
            
            # Should not raise error
            assert x.shape == (1, 1, 224, 224)
            
            Path(f.name).unlink(missing_ok=True)

    def test_very_loud_audio(self):
        """Test with very loud audio (clipping prevented)."""
        sr = TARGET_SR
        duration = 7.0
        t = np.linspace(0, duration, int(sr * duration))
        # Very loud audio
        audio = (10 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            
            x = audio_to_mel_tensor(f.name)
            
            # Should still be normalized 0-1
            assert x.min().item() >= 0.0
            assert x.max().item() <= 1.0
            
            Path(f.name).unlink(missing_ok=True)

    def test_short_audio_file(self):
        """Test with audio shorter than FIXED_LEN."""
        sr = TARGET_SR
        duration = 2.0  # Much shorter than 7 seconds
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            
            x = audio_to_mel_tensor(f.name)
            
            # Should still produce (1, 1, 224, 224)
            assert x.shape == (1, 1, 224, 224)
            
            Path(f.name).unlink(missing_ok=True)

    def test_long_audio_file(self):
        """Test with audio longer than FIXED_LEN."""
        sr = TARGET_SR
        duration = 15.0  # Much longer than 7 seconds
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            
            x = audio_to_mel_tensor(f.name)
            
            # Should still produce (1, 1, 224, 224)
            assert x.shape == (1, 1, 224, 224)
            
            Path(f.name).unlink(missing_ok=True)


class TestPreprocessingConstants:
    """Test that preprocessing constants are correctly defined."""

    def test_target_sr_is_positive(self):
        """Test that TARGET_SR is positive."""
        assert TARGET_SR > 0
        assert TARGET_SR == 8000

    def test_fixed_len_is_positive(self):
        """Test that FIXED_LEN is positive."""
        assert FIXED_LEN > 0
        assert FIXED_LEN == 56000  # 8000 * 7.0

    def test_n_mels_is_positive(self):
        """Test that N_MELS is positive."""
        assert N_MELS > 0
        assert N_MELS == 128

    def test_n_fft_is_positive(self):
        """Test that N_FFT is positive."""
        assert N_FFT > 0
        assert N_FFT == 1024

    def test_hop_length_is_positive(self):
        """Test that HOP_LENGTH is positive."""
        assert HOP_LENGTH > 0
        assert HOP_LENGTH == 256

    def test_hop_length_less_than_n_fft(self):
        """Test that HOP_LENGTH < N_FFT (standard practice)."""
        assert HOP_LENGTH < N_FFT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
