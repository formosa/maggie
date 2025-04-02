# -*- coding: utf-8 -*-
"""
Test script for kokoro_onnx TTS functionality.

"""

performance_times = {
   "import soundfile": None,
   "import sounddevice": None,
   "import kokoro_onnyx": None,
   "from misaki import en, espeak": None,
   "import pyaudio": None,
   "import numpy": None,
   "koroko instance created (text)": None,
   "koroko instance created (phoneme)": None,
   "t_samples, t_sample_rate": None,
   "sd.play (text)": None,
   "sd.wait (text)": None,
   "play_audio function": None,
   "play_audio": None,
   "G2P instance": None,
   "phoneme conversion": None,
   "phonemes": None,
   "kokoro instance created": None,
   "p_samples, p_sample_rate": None,
   "sd.play (phonemes)": None,
   "sd.wait (phonemes)": None,
   "play_audio (phonemes)": None
}
"""dict: Dictionary storing performance timing data for module imports.

   Each key represents an import statement being timed, and each value
   stores the execution time (in seconds) or None if not yet measured.

   Attributes
   ----------
   "import soundfile" : float or None
      Time taken to import the soundfile module
   "import sounddevice" : float or None
      Time taken to import the sounddevice module
   "import pyaudio" : float or None
      Time taken to import the pyaudio module
   "import kokoro_onnyx" : float or None
      Time taken to import the kokoro_onnyx module
   "from misaki import en, espeak" : float or None
      Time taken to import the en and espeak modules from misaki
   "koroko instance created (text)" : float or None
      Time taken to create an instance of the Kokoro class
   "t_samples, t_sample_rate" : float or None
      Time taken to create audio samples and sample rate using Kokoro
   "sd.play (text)" : float or None
      Time taken to play audio samples using sounddevice
   "sd.wait (text)" : float or None
      Time taken to wait for audio playback to finish using sounddevice
   "play_audio function" : float or None
      Time taken to create the play_audio function using pyaudio
   "play_audio" : float or None
      Time taken to play audio samples using the play_audio function
   "G2P instance" : float or None
      Time taken to create an instance of the G2P class from misaki
   "phoneme conversion" : float or None
      Time taken to perform phoneme conversion using the G2P instance
   "phonemes" : list or None
      List of phonemes generated from the input text
   "kokoro instance created" : float or None
      Time taken to create an instance of the Kokoro class
   "p_samples, p_sample_rate" : float or None
      Time taken to create audio samples and sample rate using Kokoro with phonemes
   "sd.play (phonemes)" : float or None
      Time taken to play audio samples using sounddevice with phonemes
   "sd.wait (phonemes)" : float or None
      Time taken to wait for audio playback to finish using sounddevice with phonemes
   "play_audio (phonemes)" : float or None
      Time taken to play audio samples using the play_audio function with phonemes

   Examples
   --------
   >>> print(performance_times["import soundfile"])
   0.02345
"""

# Untracked imports
import time
import os

# soundfile can be used to save audio files in various formats.
# soundfile is a library for reading and writing sound files in Python.
start_time = time.perf_counter()
import soundfile as sf
performance_times["import soundfile"] = time.perf_counter() - start_time
print(f'soundfile imported in: {performance_times["import soundfile"]:.5f} seconds')

# sounddevice is used for audio playback
# sounddevice is a library for playing and recording sound in Python.
start_time = time.perf_counter()
import sounddevice as sd
performance_times["import sounddevice"] = time.perf_counter() - start_time
print(f'sounddevice imported in: {performance_times["import sounddevice"]:.5f} seconds')

# pyaudio is used for audio playback
# pyaudio is a library for playing and recording sound in Python.
start_time = time.perf_counter()
import pyaudio
performance_times["import pyaudio"] = time.perf_counter() - start_time
print(f'pyaudio imported in: {performance_times["import pyaudio"]:.5f} seconds')

# kokoro_onnx is a library for text-to-speech synthesis using ONNX models.
# It provides a simple interface for generating speech from text using pre-trained models.
start_time = time.perf_counter()
from kokoro_onnx import Kokoro
performance_times["import kokoro_onnx"] = time.perf_counter() - start_time
print(f'kokoro_onnx imported in: {performance_times["import kokoro_onnx"]:.5f} seconds')

# misaki is a library for text-to-speech synthesis and phoneme conversion.
# It includes various modules for different languages and functionalities.
start_time = time.perf_counter()
from misaki import en, espeak
performance_times["from misaki import en, espeak"] = time.perf_counter() - start_time
print(f'misaki imported in: {performance_times["from misaki import en, espeak"]:.5f} seconds')

# numpy is a library for numerical computations in Python.
# It provides support for arrays, matrices, and a wide range of mathematical functions.
start_time = time.perf_counter()
import numpy as np
performance_times["import numpy"] = time.perf_counter() - start_time
print(f'numpy imported in: {performance_times["import numpy"]:.5f} seconds')


# Kokoro - fixed path handling using os.path.join and raw strings
model_dir = "C:\\AI\\claude\\service\\maggie\\maggie\\models\\tts"
kokoro_model = os.path.join(model_dir, "kokoro-v1.0.onnx")
kokoro_weights = os.path.join(model_dir, "voices-v1.0.bin")
print(f"kokoro_model: {kokoro_model}")
print(f"kokoro_weights (voices): {kokoro_weights}")

# Kokoro - instance creation with espeak fallback
# The Kokoro class is used to create a TTS instance with specified model and weights.
start_time = time.perf_counter()
kokoro = Kokoro(kokoro_model, kokoro_weights)
performance_times["koroko instance created (text)"] = time.perf_counter() - start_time
print(f'kokoro instance created in: {performance_times["koroko instance created (text)"]:.5f} seconds')

text_to_speech = """
Hello, my name is maggie!
I am your new A.I. personal assistant.
I utilize the [Misaki] G2P engine 
and cutting edge [Kokoro] models
to generate speech from text.
"""
print(f"text_to_speech: {text_to_speech}")

# Kokoro - text-to-speech synthesis without phoneme conversion
start_time = time.perf_counter()
(t_samples, t_sample_rate) = kokoro.create(
    text=text_to_speech, 
    voice="af_heart", 
    speed=1.25, 
    lang="en-us",
    is_phonemes=False,
    trim=True
)
performance_times["(t_samples, t_sample_rate)"] = time.perf_counter() - start_time
print(f'(t_samples, t_sample_rate) (is_phonemes=False) created in: {performance_times["(t_samples, t_sample_rate)"]:.5f} seconds')

# sounddevice - play audio samples
start = time.perf_counter()
sd.play(t_samples, t_sample_rate)
performance_times["sd.play (text)"] = time.perf_counter() - start
print(f'sd.play(t_samples, t_sample_rate) ended in: {performance_times["sd.play (text)"]:.5f} seconds')

# sounddevice - wait for audio playback to finish
# sd.wait() blocks the program until the audio playback is finished.
start = time.perf_counter()
sd.wait()
performance_times["sd.wait (text)"] = time.perf_counter() - start
print(f'sd.wait() ended in: {performance_times["sd.wait (text)"]:.5f} seconds')


# The play_audio function uses pyaudio to play audio samples.
start = time.perf_counter()
def play_audio(t_samples, t_sample_rate)->None:
   
   audio_int16 = np.clip(
      t_samples*32767,
      -32768,
      32767
   ).astype(np.int16)

   p=pyaudio.PyAudio()
   stream=p.open(
       format=pyaudio.paInt16,
       channels=1,
       rate=t_sample_rate,
       output=True,
       frames_per_buffer=512
   )

   chunk_size=512
   for i in range(0,len(audio_int16),chunk_size):
       chunk=audio_int16[i:i+chunk_size].tobytes()
       stream.write(chunk)
   stream.stop_stream()
   stream.close()
   p.terminate()

performance_times["play_audio function"] = time.perf_counter() - start
print(f'play_audio function created in: {performance_times["play_audio function"]:.5f} seconds')

# PyAudio - play audio samples
start = time.perf_counter()
play_audio(t_samples, t_sample_rate)
performance_times["play_audio (text)"] = time.perf_counter() - start
print(f'play_audio(t_samples, t_sample_rate) ended in: {performance_times["play_audio (text)"]:.5f} seconds')


# Misaki G2P with espeak-ng fallback
# The G2P class is used for grapheme-to-phoneme conversion.
start_time = time.perf_counter()
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)
performance_times["G2P instance"] = time.perf_counter() - start_time
print(f'G2P instance created in: {performance_times["G2P instance"]:.5f} seconds')

# Misaki - phoneme conversion
start_time = time.perf_counter()
from kokoro_onnx import EspeakConfig
text_to_speech = "(/misˈɑki/) (/kˈOkəɹO/)"
phonemes, _ = g2p(text_to_speech)
performance_times["phoneme conversion"] = time.perf_counter() - start_time
print(f'phoneme conversion ended in: {performance_times["phoneme conversion"]:.5f} seconds')
print(f'phonemes: {phonemes}')


# Kokoro - instance creation with espeak fallback
# The Kokoro class is used to create a TTS instance with specified model and weights.
start_time = time.perf_counter()
kokoro = Kokoro(kokoro_model, kokoro_weights, espeak_config= EspeakConfig(lib_path='venv\Lib\site-packages\espeakng_loader\espeak-ng.dll', data_path='venv\Lib\site-packages\espeakng_loader\espeak-ng-data') )
performance_times["koroko instance created (phoneme)"] = time.perf_counter() - start_time
print(f'kokoro instance created in: {performance_times["koroko instance created (phoneme)"]:.5f} seconds')


# Kokoro - text-to-speech synthesis with phoneme conversion
start_time = time.perf_counter()
p_samples, p_sample_rate = kokoro.create(
    text=phonemes, 
    voice="af_heart", 
    speed=1.25, 
    lang="en-us",
    is_phonemes=True,
    trim=True
)
performance_times["(p_samples, p_sample_rate)"] = time.perf_counter() - start_time
print(f'(p_samples, p_sample_rate) created in: {performance_times["(p_samples, p_sample_rate)"]:.5f} seconds')

# PyAudio - play audio samples with phoneme conversion
start = time.perf_counter()
play_audio(p_samples, p_sample_rate)
performance_times["play_audio (phonemes)"] = time.perf_counter() - start
print(f'play_audio(p_samples, p_sample_rate) ended in: {performance_times["play_audio (phonemes)"]:.5f} seconds')

# sounddevice - play audio samples with phoneme conversion
start = time.perf_counter()
sd.play(p_samples, p_sample_rate)
performance_times["sd.play (phonemes)"] = time.perf_counter() - start
print(f'sd.play(p_samples, p_sample_rate) ended in: {performance_times["sd.play (phonemes)"]:.5f} seconds')

# sounddevice - wait for audio playback to finish with phoneme conversion
start = time.perf_counter()
sd.wait()
performance_times["sd.wait (phonemes)"] = time.perf_counter() - start
print(f'sd.wait() (phonemes) ended in: {performance_times["sd.wait (phonemes)"]:.5f} seconds')
print(f"""
=============================== [ RESULTS ]=====================================

LIBRARY IMPORT TIMES:
---------------------------------------------------------------
import soundfile: {performance_times["import soundfile"]:.5f} seconds

{"* " if performance_times["import sounddevice"] < performance_times["import pyaudio"] else ""}import sounddevice: {performance_times["import sounddevice"]:.5f} seconds
{"* " if performance_times["import pyaudio"] < performance_times["import sounddevice"] else ""}import pyaudio: {performance_times["import pyaudio"]:.5f} seconds

import kokoro_onnx: {performance_times["import kokoro_onnx"]:.5f} seconds

import misaki: {performance_times["from misaki import en, espeak"]:.5f} seconds

import numpy: {performance_times["import numpy"]:.5f} seconds

PHONEME CONVERSION TIMES:
---------------------------------------------------------------
G2P instance created in: {performance_times["G2P instance"]:.5f} seconds
phoneme conversion ended in: {performance_times["phoneme conversion"]:.5f} seconds

Total time for phoneme creation: {performance_times["G2P instance"] + performance_times["phoneme conversion"]:.5f} seconds
phonemes: {phonemes}

KOKORO INSTANCE TIMES:
---------------------------------------------------------------
TEXT:
kokoro instance created in: {performance_times["koroko instance created (text)"]:.5f} seconds

PHONEMES:
kokoro instance created in: {performance_times["koroko instance created (phoneme)"]:.5f} seconds

SOUND PLAYBACK TIMES:
---------------------------------------------------------------
TEXT:
{"* " if performance_times["sd.play (text)"] < performance_times["play_audio (text)"] else ""}sd.play(t_samples, t_sample_rate) ended in: {performance_times["sd.play (text)"]:.5f} seconds
{"* " if performance_times["play_audio (text)"] < performance_times["sd.play (text)"] else ""}play_audio(t_samples, t_sample_rate) ended in: {performance_times["play_audio (text)"]:.5f} seconds
sd.wait() ended in: {performance_times["sd.wait (text)"]:.5f} seconds

PHONEMES:
{"* " if performance_times["sd.play (phonemes)"] < performance_times["play_audio (phonemes)"] else ""}sd.play(p_samples, p_sample_rate) ended in: {performance_times["sd.play (phonemes)"]:.5f} seconds
{"* " if performance_times["play_audio (phonemes)"] < performance_times["sd.play (phonemes)"] else ""}play_audio(p_samples, p_sample_rate) ended in: {performance_times["play_audio (phonemes)"]:.5f} seconds
sd.wait() (phonemes) ended in: {performance_times["sd.wait (phonemes)"]:.5f} seconds
      
""")