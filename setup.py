from setuptools import setup, find_packages

setup(
    name="maggie",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pvporcupine==2.2.0",
        "SpeechRecognition==3.10.0",
        "PyAudio==0.2.13",
        "faster-whisper==0.9.0",
        "piper-tts==1.2.0", 
        "ctransformers==0.2.27",
        "llama-cpp-python==0.2.11",
        "PyQt6==6.5.0",
        "python-docx==0.8.11",
        "transitions==0.9.0",
        "numpy==1.24.2",
        "PyYAML==6.0",
        "loguru==0.7.0",
        "soundfile==0.12.1",
        "requests==2.31.0",
        "tqdm==4.66.1",
        "psutil==5.9.5",
        "mammoth==1.7.0",
    ],
    extras_require={
        "gpu": [
            "torch==2.0.1+cu118",
            "onnxruntime-gpu==1.15.1",
        ],
    },
    python_requires=">=3.10, <3.11",
)