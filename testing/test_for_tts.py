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


import time

start = time.perf_counter()

import soundfile as sf
import os

import sounddevice as sd
from kokoro_onnx import Kokoro


end = time.perf_counter()
print(f"Libraries Imported in: {end - start:.5f} seconds")
start = time.perf_counter()

# Misaki G2P with espeak-ng fallback
# fallback = espeak.EspeakFallback(british=False)
# g2p = en.G2P(trf=False, british=False, fallback=fallback)


# Kokoro - fixed path handling using os.path.join and raw strings
model_dir = r"C:\\AI\\claude\\service\\maggie\\maggie\\models\\tts"
kokoro_model = os.path.join(model_dir, "kokoro-v1.0.onnx")
kokoro_weights = os.path.join(model_dir, "voices-v1.0.bin")
kokoro = Kokoro(kokoro_model, kokoro_weights)


end = time.perf_counter()
print(f"kokoro Instance Created in: {end - start:.5f} seconds")
start = time.perf_counter()

samples, sample_rate = kokoro.create(
    "Hello, my name is maggie!  How can I assist you today?", voice="af_heart", speed=1.25, lang="en-us"
)

end = time.perf_counter()
print(f"samples, sample_rate Returned in: {end - start:.5f} seconds")
start = time.perf_counter()

print("Playing audio...")
sd.play(samples, sample_rate)

end = time.perf_counter()
print(f"sd.play(samples, sample_rate) ended in: {end - start:.5f} seconds")
start = time.perf_counter()

sd.wait()

print("Audio playback finished.")

end = time.perf_counter()
print(f"sd.wait() ended in: {end - start:.5f} seconds")
start = time.perf_counter()


import numpy as np
import pyaudio

def play_audio(samples, sample_rate)->None:
   audio_int16=np.clip(samples*32767,-32768,32767).astype(np.int16)
   p=pyaudio.PyAudio()
   stream=p.open(format=pyaudio.paInt16,channels=1,rate=sample_rate,output=True,frames_per_buffer=512)
   chunk_size=512
   for i in range(0,len(audio_int16),chunk_size):
       chunk=audio_int16[i:i+chunk_size].tobytes();stream.write(chunk)
   stream.stop_stream();stream.close();p.terminate()


end = time.perf_counter()
print(f"PyAudio Setup loaded in: {end - start:.5f} seconds")
start = time.perf_counter()

print("Playing audio with pyaudio...")
play_audio(samples, sample_rate)
print("Audio playback finished.")

end = time.perf_counter()
print(f"Play ended in: {end - start:.5f} seconds")