# -*- coding: utf-8 -*-
import soundfile as sf
from kokoro_onnx import Kokoro
from misaki import en, espeak
import os

# Misaki G2P with espeak-ng fallback
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)

# Kokoro
kokoro_model = os.path("C:\AI\claude\service\maggie\maggie\models\tts\kokoro-v1.0.onnx")
kokoro_weights = os.path("C:\AI\claude\service\maggie\maggie\models\tts\voices-v1.0.bin")
kokoro = Kokoro(kokoro_model, kokoro_weights)

# Text
text = "[Misaki](/misˈɑki/) is a G2P engine designed for [Kokoro](/kˈOkəɹO/) models."
print("Text:", text)

# Phonemize
phonemes, _ = g2p(text)
print("Phonemes:", phonemes)

# Create
samples, sample_rate = kokoro.create(phonemes, "af_heart", is_phonemes=True)

# Save
sf.write("audio.wav", samples, sample_rate)


print("Created audio.wav")

