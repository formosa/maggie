# -*- coding: utf-8 -*-
"""
Test script for kokoro_onnx TTS functionality.

If you encounter UnicodeDecodeError, try one of these solutions:
1. Set the PYTHONIOENCODING environment variable to utf-8:
   - On Windows: set PYTHONIOENCODING=utf-8
   - On Linux/Mac: export PYTHONIOENCODING=utf-8
2. Reinstall the kokoro-onnx package:
   - pip uninstall kokoro-onnx
   - pip install kokoro-onnx --no-cache-dir
"""
import soundfile as sf
import os
import sys

try:
    from kokoro_onnx import Kokoro
    from misaki import en, espeak
except UnicodeDecodeError as e:
    print("\nUnicodeDecodeError when importing kokoro_onnx:")
    print(e)
    print("\nThis error occurs because the kokoro_onnx package contains files with")
    print("non-ASCII characters that Python can't decode with the default encoding.")
    print("\nTo fix this, try one of these solutions:")
    print("1. Set the PYTHONIOENCODING environment variable to utf-8:")
    print("   - On Windows: set PYTHONIOENCODING=utf-8")
    print("   - On Linux/Mac: export PYTHONIOENCODING=utf-8")
    print("2. Reinstall the kokoro-onnx package:")
    print("   - pip uninstall kokoro-onnx")
    print("   - pip install kokoro-onnx --no-cache-dir")
    sys.exit(1)

# Misaki G2P with espeak-ng fallback
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)

# Kokoro - fixed path handling using os.path.join and raw strings
model_dir = r"C:\AI\claude\service\maggie\maggie\models\tts"
kokoro_model = os.path.join(model_dir, "kokoro-v1.0.onnx")
kokoro_weights = os.path.join(model_dir, "voices-v1.0.bin")

# Check if files exist
if not os.path.exists(kokoro_model):
    print(f"Error: Model file not found at {kokoro_model}")
    sys.exit(1)
if not os.path.exists(kokoro_weights):
    print(f"Error: Weights file not found at {kokoro_weights}")
    sys.exit(1)

# Initialize Kokoro
try:
    kokoro = Kokoro(kokoro_model, kokoro_weights)
    print("Successfully initialized Kokoro TTS engine")
except Exception as e:
    print(f"Error initializing Kokoro: {e}")
    sys.exit(1)

# Text to synthesize
text = "[Misaki](/misˈɑki/) is a G2P engine designed for [Kokoro](/kˈOkəɹO/) models."
print("Text:", text)

# Phonemize
try:
    phonemes, _ = g2p(text)
    print("Phonemes:", phonemes)
except Exception as e:
    print(f"Error during phonemization: {e}")
    sys.exit(1)

# Create audio
try:
    samples, sample_rate = kokoro.create(phonemes, "af_heart", is_phonemes=True)
except Exception as e:
    print(f"Error creating audio: {e}")
    sys.exit(1)

# Save audio file
try:
    sf.write("audio.wav", samples, sample_rate)
    print("Created audio.wav")
except Exception as e:
    print(f"Error saving audio file: {e}")
    sys.exit(1)