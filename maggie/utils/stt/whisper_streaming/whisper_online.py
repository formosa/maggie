#!/usr/bin/env python3
"""
Whisper Online Module for Real-time Audio Transcription.

This module provides classes and functions for real-time audio transcription
using various Whisper model backends. It includes ASR base classes, 
processing components, and utility functions.

The module supports multiple ASR backends, Voice Activity Detection (VAD),
and various buffer management strategies for optimized transcription.
"""

import sys
import numpy as np
import librosa
from functools import lru_cache
import time
import logging
import io
import soundfile as sf
import math

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")


@lru_cache(10**6)
def load_audio(fname):
    """
    Load an audio file into memory with caching.
    
    Parameters
    ----------
    fname : str
        Path to the audio file.
        
    Returns
    -------
    numpy.ndarray
        Audio data as a float32 array sampled at 16kHz.
        
    Notes
    -----
    Results are cached using lru_cache for efficiency when accessing
    the same file multiple times.
    """
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a


def load_audio_chunk(fname, beg, end):
    """
    Load a specific chunk of an audio file.
    
    Parameters
    ----------
    fname : str
        Path to the audio file.
    beg : float
        Beginning time in seconds.
    end : float
        Ending time in seconds.
        
    Returns
    -------
    numpy.ndarray
        Audio chunk data as a float32 array.
        
    Notes
    -----
    This function leverages the caching of load_audio() for efficiency.
    """
    audio = load_audio(fname)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


def create_tokenizer(lan):
    """
    Create a sentence tokenizer for a specific language.
    
    Parameters
    ----------
    lan : str
        Language code that must be in WHISPER_LANG_CODES.
        
    Returns
    -------
    object
        A tokenizer object with a 'split' method for sentence segmentation.
        
    Raises
    ------
    AssertionError
        If the language code is not supported.
        
    Notes
    -----
    Depending on the language, uses MosesTokenizer, WtP, or a custom tokenizer.
    """
    assert lan in WHISPER_LANG_CODES, "Language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    if lan == "uk":
        import tokenize_uk
        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)
        return UkrainianTokenizer()

    # Supported by fast-mosestokenizer
    if lan in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split():
        from mosestokenizer import MosesTokenizer
        return MosesTokenizer(lan)

    # Languages not supported by wtpsplit
    if lan in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split():
        logger.debug(f"{lan} code is not supported by wtpsplit. Going to use None lang_code option.")
        lan = None

    from wtpsplit import WtP
    # Downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")
    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)
    return WtPtok()


def add_shared_args(parser):
    """
    Add shared command-line arguments to an argument parser.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to add arguments to.
        
    Notes
    -----
    Adds arguments for model size, language, ASR backend, chunk size,
    buffer trimming, VAD/VAC settings, and logging.
    """
    parser.add_argument('--min-chunk-size', type=float, default=1.0, 
                        help='Minimum audio chunk size in seconds. It waits up to this time to do processing.')
    parser.add_argument('--model', type=str, default='large-v2', 
                        choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo".split(","),
                        help="Name size of the Whisper model to use (default: large-v2).")
    parser.add_argument('--model_cache_dir', type=str, default=None, 
                        help="Overriding the default model cache dir where models downloaded from the hub are saved")
    parser.add_argument('--model_dir', type=str, default=None, 
                        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
    parser.add_argument('--lan', '--language', type=str, default='auto', 
                        help="Source language code, e.g. en,de,cs, or 'auto' for language detection.")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe","translate"],
                        help="Transcribe or translate.")
    parser.add_argument('--backend', type=str, default="faster-whisper", 
                        choices=["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api"],
                        help='Load only this backend for Whisper processing.')
    parser.add_argument('--vac', action="store_true", default=False, 
                        help='Use VAC = voice activity controller. Recommended. Requires torch.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, 
                        help='VAC sample size in seconds.')
    parser.add_argument('--vad', action="store_true", default=False, 
                        help='Use VAD = voice activity detection, with the default parameters.')
    parser.add_argument('--buffer_trimming', type=str, default="segment", 
                        choices=["sentence", "segment"],
                        help='Buffer trimming strategy -- trim completed sentences or segments.')
    parser.add_argument('--buffer_trimming_sec', type=float, default=15, 
                        help='Buffer trimming length threshold in seconds.')
    parser.add_argument("-l", "--log-level", dest="log_level", 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help="Set the log level", default='DEBUG')


def set_logging(args, logger, other="_server"):
    """
    Configure logging for the whisper_streaming package.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing log_level.
    logger : logging.Logger
        Logger instance to configure.
    other : str, optional
        Additional string for logger name, default is "_server".
    """
    logging.basicConfig(format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online" + other).setLevel(args.log_level)


# ASR Base Classes

class ASRBase:
    """
    Abstract base class for ASR (Automatic Speech Recognition) implementations.
    
    This class defines the common interface for all ASR backend implementations.
    Concrete subclasses must implement the required methods.
    
    Parameters
    ----------
    lan : str
        Language code for transcription.
    modelsize : str, optional
        Size of the model (e.g., 'tiny', 'base', 'small', 'medium', 'large').
    cache_dir : str, optional
        Directory for caching models.
    model_dir : str, optional
        Directory containing a pre-downloaded model.
    logfile : file, optional
        File for logging, default is sys.stderr.
        
    Attributes
    ----------
    sep : str
        Character used to join transcribed words.
    """

    sep = " "   # join transcribe words with this character (" " for whisper_timestamped,
                # "" for faster-whisper because it emits the spaces when neeeded)

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        """
        Initialize the ASR base class.
        
        Parameters
        ----------
        lan : str
            Language code for transcription.
        modelsize : str, optional
            Size of the model (e.g., 'tiny', 'base', 'small', 'medium', 'large').
        cache_dir : str, optional
            Directory for caching models.
        model_dir : str, optional
            Directory containing a pre-downloaded model.
        logfile : file, optional
            File for logging, default is sys.stderr.
        """
        self.logfile = logfile

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize, cache_dir, model_dir):
        """
        Load the ASR model.
        
        This method must be implemented by concrete subclasses.
        
        Parameters
        ----------
        modelsize : str
            Size of the model to load.
        cache_dir : str or None
            Directory for caching models.
        model_dir : str or None
            Directory containing a pre-downloaded model.
            
        Returns
        -------
        model
            The loaded model object.
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        """
        Transcribe audio with optional initial prompt.
        
        This method must be implemented by concrete subclasses.
        
        Parameters
        ----------
        audio : numpy.ndarray
            Audio data as float32 array.
        init_prompt : str, optional
            Initial prompt to guide transcription.
            
        Returns
        -------
        object
            Transcription result object.
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented in the child class")

    def use_vad(self):
        """
        Enable Voice Activity Detection.
        
        This method must be implemented by concrete subclasses.
        
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented in the child class")


class WhisperTimestampedASR(ASRBase):
    """
    ASR implementation using whisper_timestamped library.
    
    This class provides timestamped transcription using the whisper_timestamped backend.
    Initially tested with this backend, but slower than faster-whisper.
    
    Parameters
    ----------
    lan : str
        Language code for transcription.
    modelsize : str, optional
        Size of the model (e.g., 'tiny', 'base', 'small', 'medium', 'large').
    cache_dir : str, optional
        Directory for caching models.
    model_dir : str, optional
        Directory containing a pre-downloaded model.
    logfile : file, optional
        File for logging, default is sys.stderr.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        """
        Load the whisper_timestamped model.
        
        Parameters
        ----------
        modelsize : str, optional
            Size of the model to load.
        cache_dir : str or None, optional
            Directory for caching models.
        model_dir : str or None, optional
            Directory containing a pre-downloaded model.
            
        Returns
        -------
        model
            The loaded Whisper model object.
        """
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logger.debug("Ignoring model_dir, not implemented")
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        """
        Transcribe audio with optional initial prompt.
        
        Parameters
        ----------
        audio : numpy.ndarray
            Audio data as float32 array.
        init_prompt : str, optional
            Initial prompt to guide transcription.
            
        Returns
        -------
        dict
            Transcription result dictionary.
        """
        result = self.transcribe_timestamped(self.model,
                audio, language=self.original_language,
                initial_prompt=init_prompt, verbose=None,
                condition_on_previous_text=True, **self.transcribe_kargs)
        return result
 
    def ts_words(self, r):
        """
        Convert transcription result to timestamped words.
        
        Parameters
        ----------
        r : dict
            Transcription result from transcribe method.
            
        Returns
        -------
        list
            List of tuples (start_time, end_time, text).
        """
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                t = (w["start"], w["end"], w["text"])
                o.append(t)
        return o

    def segments_end_ts(self, res):
        """
        Get end timestamps of segments.
        
        Parameters
        ----------
        res : dict
            Transcription result from transcribe method.
            
        Returns
        -------
        list
            List of segment end timestamps.
        """
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        """
        Enable Voice Activity Detection.
        
        Sets the VAD flag in transcribe_kargs.
        """
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self):
        """
        Set the task to translation instead of transcription.
        
        Updates the transcribe_kargs to use the translation task.
        """
        self.transcribe_kargs["task"] = "translate"


class FasterWhisperASR(ASRBase):
    """
    ASR implementation using faster-whisper library.
    
    This class provides transcription using the faster-whisper backend,
    which offers significantly improved performance (approximately 4x faster).
    For GPU usage, requires specific CUDNN version.
    
    Parameters
    ----------
    lan : str
        Language code for transcription.
    modelsize : str, optional
        Size of the model (e.g., 'tiny', 'base', 'small', 'medium', 'large').
    cache_dir : str, optional
        Directory for caching models.
    model_dir : str, optional
        Directory containing a pre-downloaded model.
    logfile : file, optional
        File for logging, default is sys.stderr.
    """

    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        """
        Load the faster-whisper model.
        
        Parameters
        ----------
        modelsize : str, optional
            Size of the model to load.
        cache_dir : str or None, optional
            Directory for caching models.
        model_dir : str or None, optional
            Directory containing a pre-downloaded model.
            
        Returns
        -------
        model
            The loaded faster-whisper model object.
            
        Raises
        ------
        ValueError
            If neither modelsize nor model_dir is provided.
        """
        from faster_whisper import WhisperModel
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        # This worked fast and reliably on NVIDIA L40
        model = WhisperModel(model_size_or_path, device="cuda", compute_type="float16", download_root=cache_dir)

        # or run on GPU with INT8
        # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
        #model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

        # or run on CPU with INT8
        # tested: works, but slow, appx 10-times than cuda FP16
        #model = WhisperModel(modelsize, device="cpu", compute_type="int8")
        return model

    def transcribe(self, audio, init_prompt=""):
        """
        Transcribe audio with optional initial prompt.
        
        Parameters
        ----------
        audio : numpy.ndarray
            Audio data as float32 array.
        init_prompt : str, optional
            Initial prompt to guide transcription.
            
        Returns
        -------
        list
            List of segment objects.
        """
        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(audio, language=self.original_language, initial_prompt=init_prompt, beam_size=5, word_timestamps=True, condition_on_previous_text=True, **self.transcribe_kargs)
        #print(info)  # info contains language detection result

        return list(segments)

    def ts_words(self, segments):
        """
        Convert transcription segments to timestamped words.
        
        Parameters
        ----------
        segments : list
            Segments from transcribe method.
            
        Returns
        -------
        list
            List of tuples (start_time, end_time, text).
        """
        o = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        """
        Get end timestamps of segments.
        
        Parameters
        ----------
        res : list
            Segments from transcribe method.
            
        Returns
        -------
        list
            List of segment end timestamps.
        """
        return [s.end for s in res]

    def use_vad(self):
        """
        Enable Voice Activity Detection filtering.
        
        Sets the vad_filter flag in transcribe_kargs.
        """
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        """
        Set the task to translation instead of transcription.
        
        Updates the transcribe_kargs to use the translation task.
        """
        self.transcribe_kargs["task"] = "translate"


class MLXWhisper(ASRBase):
    """
    ASR implementation using MLX Whisper library for Apple Silicon.
    
    This class provides optimized transcription for Apple Silicon processors.
    Significantly faster than faster-whisper (without CUDA) on Apple M1.
    
    Parameters
    ----------
    lan : str
        Language code for transcription.
    modelsize : str, optional
        Size of the model (e.g., 'tiny', 'base', 'small', 'medium', 'large').
    cache_dir : str, optional
        Directory for caching models (not used by MLX Whisper).
    model_dir : str, optional
        Directory containing a pre-downloaded model.
    logfile : file, optional
        File for logging, default is sys.stderr.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        """
        Load the MLX Whisper model.
        
        Parameters
        ----------
        modelsize : str, optional
            Size of the model to load.
        cache_dir : str or None, optional
            Directory for caching models.
        model_dir : str or None, optional
            Directory containing a pre-downloaded model.
            
        Returns
        -------
        function
            The transcribe function from MLX Whisper.
        """
        from mlx_whisper.transcribe import ModelHolder, transcribe
        import mlx.core as mx # Is installed with mlx-whisper
        
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize parameter is not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = self.translate_model_name(modelsize)
            logger.debug(f"Loading whisper model {modelsize}. You use mlx whisper, so {model_size_or_path} will be used.")
        
        self.model_size_or_path = model_size_or_path
        
        # Note: ModelHolder.get_model loads the model into a static class variable, 
        # making it a global resource. This means:
        # - Only one model can be loaded at a time; switching models requires reloading.
        # - This approach may not be suitable for scenarios requiring multiple models simultaneously,
        #   such as using whisper-streaming as a module with varying model sizes.
        dtype = mx.float16 # Default to mx.float16
        ModelHolder.get_model(model_size_or_path, dtype) #Model is preloaded to avoid reloading during transcription
        
        return transcribe
    
    def translate_model_name(self, model_name):
        """
        Translates a given model name to its corresponding MLX-compatible model path.
        
        Parameters
        ----------
        model_name : str
            The name of the model to translate.
            
        Returns
        -------
        str
            The MLX-compatible model path.
            
        Raises
        ------
        ValueError
            If the model name is not recognized or not supported.
        """
        # Dictionary mapping model names to MLX-compatible paths
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx"
        }

        # Retrieve the corresponding MLX model path
        mlx_model_path = model_mapping.get(model_name)

        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")
    
    def transcribe(self, audio, init_prompt=""):
        """
        Transcribe audio with optional initial prompt.
        
        Parameters
        ----------
        audio : numpy.ndarray
            Audio data as float32 array.
        init_prompt : str, optional
            Initial prompt to guide transcription.
            
        Returns
        -------
        list
            List of segment dictionaries.
        """
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
            **self.transcribe_kargs
        )
        return segments.get("segments", [])

    def ts_words(self, segments):
        """
        Extract timestamped words from segments.
        
        Parameters
        ----------
        segments : list
            Segments from transcribe method.
            
        Returns
        -------
        list
            List of tuples (start_time, end_time, text).
        """
        return [
            (word["start"], word["end"], word["word"])
            for segment in segments
            for word in segment.get("words", [])
            if segment.get("no_speech_prob", 0) <= 0.9
        ]
    
    def segments_end_ts(self, res):
        """
        Get end timestamps of segments.
        
        Parameters
        ----------
        res : list
            Segments from transcribe method.
            
        Returns
        -------
        list
            List of segment end timestamps.
        """
        return [s['end'] for s in res]

    def use_vad(self):
        """
        Enable Voice Activity Detection filtering.
        
        Sets the vad_filter flag in transcribe_kargs.
        """
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        """
        Set the task to translation instead of transcription.
        
        Updates the transcribe_kargs to use the translation task.
        """
        self.transcribe_kargs["task"] = "translate"


class OpenaiApiASR(ASRBase):
    """
    ASR implementation using OpenAI's Whisper API.
    
    Parameters
    ----------
    lan : str, optional
        ISO-639-1 language code for transcription.
    temperature : float, optional
        Temperature for sampling. Default is 0.
    logfile : file, optional
        File for logging, default is sys.stderr.
    """

    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        """
        Initialize the OpenaiApiASR.
        
        Parameters
        ----------
        lan : str, optional
            ISO-639-1 language code for transcription.
        temperature : float, optional
            Temperature for sampling. Default is 0.
        logfile : file, optional
            File for logging, default is sys.stderr.
        """
        self.logfile = logfile

        self.modelname = "whisper-1"  
        self.original_language = None if lan == "auto" else lan # ISO-639-1 language code
        self.response_format = "verbose_json" 
        self.temperature = temperature

        self.load_model()

        self.use_vad_opt = False

        # reset the task in set_translate_task
        self.task = "transcribe"

    def load_model(self, *args, **kwargs):
        """
        Initialize the OpenAI client.
        
        Returns
        -------
        None
            Initializes the OpenAI client for API access.
        """
        from openai import OpenAI
        self.client = OpenAI()

        self.transcribed_seconds = 0  # for logging how many seconds were processed by API, to know the cost
        

    def ts_words(self, segments):
        """
        Convert API response to timestamped words.
        
        Parameters
        ----------
        segments : object
            API response from transcribe method.
            
        Returns
        -------
        list
            List of tuples (start_time, end_time, text).
        """
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                # TODO: threshold can be set from outside
                if segment["no_speech_prob"] > 0.8:
                    no_speech_segments.append((segment.get("start"), segment.get("end")))

        o = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                # print("Skipping word", word.get("word"), "because it's in a no-speech segment")
                continue
            o.append((start, end, word.word))
        return o


    def segments_end_ts(self, res):
        """
        Get end timestamps of words.
        
        Parameters
        ----------
        res : object
            API response from transcribe method.
            
        Returns
        -------
        list
            List of word end timestamps.
        """
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        """
        Transcribe audio using OpenAI API.
        
        Parameters
        ----------
        audio_data : numpy.ndarray
            Audio data as float32 array.
        prompt : str, optional
            Prompt to guide transcription.
        *args, **kwargs
            Additional arguments for OpenAI API.
            
        Returns
        -------
        object
            OpenAI API response object.
        """
        # Write the audio data to a buffer
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format='WAV', subtype='PCM_16')
        buffer.seek(0)  # Reset buffer's position to the beginning

        self.transcribed_seconds += math.ceil(len(audio_data)/16000)  # it rounds up to the whole seconds

        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"]
        }
        if self.task != "translate" and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt

        if self.task == "translate":
            proc = self.client.audio.translations
        else:
            proc = self.client.audio.transcriptions

        # Process transcription/translation
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")

        return transcript

    def use_vad(self):
        """
        Enable Voice Activity Detection filtering.
        
        Sets the use_vad_opt flag.
        """
        self.use_vad_opt = True

    def set_translate_task(self):
        """
        Set the task to translation instead of transcription.
        
        Updates the task property.
        """
        self.task = "translate"


class HypothesisBuffer:
    """
    Buffer for managing transcription hypotheses.
    
    This class maintains and processes interim transcription outputs.
    
    Parameters
    ----------
    logfile : file, optional
        File for logging, default is sys.stderr.
        
    Attributes
    ----------
    commited_in_buffer : list
        Hypotheses that have been committed but are still in the buffer
    buffer : list
        Current buffer of hypotheses
    new : list
        New hypotheses to be processed
    last_commited_time : float
        Timestamp of the last committed word
    last_commited_word : str or None
        Last committed word
    """

    def __init__(self, logfile=sys.stderr):
        """
        Initialize the HypothesisBuffer.
        
        Parameters
        ----------
        logfile : file, optional
            File for logging, default is sys.stderr.
        """
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None

        self.logfile = logfile

    def insert(self, new, offset):
        """
        Insert new transcription hypotheses with time offset.
        
        Parameters
        ----------
        new : list
            List of tuples (start_time, end_time, text).
        offset : float
            Time offset to apply to timestamps.
        """
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new
        
        new = [(a+offset, b+offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time-0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5)+1):  # 5 is the maximum 
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i+1)][::-1])
                        tail = " ".join(self.new[j-1][2] for j in range(1, i+1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        """
        Flush and return confirmed transcription chunks.
        
        Returns
        -------
        list
            List of confirmed chunks as tuples (start_time, end_time, text).
        """
        # returns commited chunk = the longest common prefix of 2 last inserts. 

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        """
        Remove committed chunks up to the specified time.
        
        Parameters
        ----------
        time : float
            Time threshold for removal.
        """
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        """
        Return the complete buffer content.
        
        Returns
        -------
        list
            List of buffer content as tuples (start_time, end_time, text).
        """
        return self.buffer


class OnlineASRProcessor:
    """
    Processor for online (real-time) audio transcription.
    
    This class manages the audio buffer and transcription process for streaming audio.
    
    Parameters
    ----------
    asr : ASRBase
        ASR implementation to use for transcription.
    tokenizer : object, optional
        Sentence tokenizer for the target language.
    buffer_trimming : tuple, optional
        Tuple of (strategy, seconds) for buffer trimming.
        Strategy can be 'sentence' or 'segment'. Default is ("segment", 15).
    logfile : file, optional
        File for logging, default is sys.stderr.
        
    Attributes
    ----------
    SAMPLING_RATE : int
        Audio sampling rate, fixed at 16000 Hz.
    """

    SAMPLING_RATE = 16000

    def __init__(self, asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr):
        """
        Initialize the OnlineASRProcessor.
        
        Parameters
        ----------
        asr : ASRBase
            ASR implementation to use for transcription.
        tokenizer : object, optional
            Sentence tokenizer for the target language. Must have a method 'split'
            that behaves like MosesTokenizer. Can be None if using "segment" buffer trimming.
        buffer_trimming : tuple, optional
            Tuple of (strategy, seconds) for buffer trimming.
            Strategy can be 'sentence' or 'segment'. Default is ("segment", 15).
        logfile : file, optional
            File for logging, default is sys.stderr.
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile

        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset=None):
        """
        Initialize or reset the processor.
        
        Parameters
        ----------
        offset : float, optional
            Initial time offset.
        """
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio):
        """
        Add an audio chunk to the buffer.
        
        Parameters
        ----------
        audio : numpy.ndarray
            Audio data as float32 array.
        """
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """
        Generate prompt from committed text.
        
        Returns
        -------
        tuple
            (prompt, context) where prompt is for ASR and context is for debugging.
        """
        k = max(0, len(self.commited)-1)
        while k > 0 and self.commited[k-1][1] > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x)+1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _, _, t in non_prompt)

    def process_iter(self):
        """
        Process the current audio buffer and return transcription.
        
        Returns
        -------
        tuple
            (start_time, end_time, text) or (None, None, "").
        """
        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(f"Transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        # Transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        completed = self.to_flush(o)
        logger.debug(f">>>>COMPLETE NOW: {completed}")
        the_rest = self.to_flush(self.transcript_buffer.complete())
        logger.debug(f"INCOMPLETE: {the_rest}")

        # There is a newly confirmed text

        if o and self.buffer_trimming_way == "sentence":  # trim the completed sentences
            if len(self.audio_buffer)/self.SAMPLING_RATE > self.buffer_trimming_sec:  # longer than this
                self.chunk_completed_sentence()

        if self.buffer_trimming_way == "segment":
            s = self.buffer_trimming_sec  # trim the completed segments longer than s,
        else:
            s = 30 # if the audio buffer is longer than 30s, trim it
        
        if len(self.audio_buffer)/self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)

            # Alternative: on any word
            #l = self.buffer_time_offset + len(self.audio_buffer)/self.SAMPLING_RATE - 10
            # Let's find commited word that is less
            #k = len(self.commited)-1
            #while k>0 and self.commited[k][1] > l:
            #    k -= 1
            #t = self.commited[k][1] 
            logger.debug("chunking segment")
            #self.chunk_at(t)

        logger.debug(f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}")
        return self.to_flush(o)

    def chunk_completed_sentence(self):
        """
        Trim the audio buffer at the end of a completed sentence.
        """
        if self.commited == []:
            return
        logger.debug(self.commited)
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            logger.debug(f"\t\tSENT: {s}")
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)
        # we will continue with audio processing at this timestamp
        chunk_at = sents[-2][1]

        logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        """
        Trim the audio buffer at the end of a completed segment.
        
        Parameters
        ----------
        res : object
            Transcription result from ASR.
        """
        if self.commited == []:
            return

        ends = self.asr.segments_end_ts(res)

        t = self.commited[-1][1]

        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                logger.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
            else:
                logger.debug(f"--- last segment not within commited area")
        else:
            logger.debug(f"--- not enough segments to chunk")

    def chunk_at(self, time):
        """
        Trim the hypothesis and audio buffer at a specific time.
        
        Parameters
        ----------
        time : float
            Time threshold for trimming.
        """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds*self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def words_to_sentences(self, words):
        """
        Convert words to sentences using the tokenizer.
        
        Parameters
        ----------
        words : list
            List of tuples (start_time, end_time, text).
            
        Returns
        -------
        list
            List of tuples (start_time, end_time, sentence).
        """
        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b, e, w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg, end, fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        """
        Finalize processing and return any remaining transcription.
        
        Returns
        -------
        tuple
            (start_time, end_time, text) or (None, None, "").
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, noncommited: {f}")
        self.buffer_time_offset += len(self.audio_buffer)/16000
        return f

    def to_flush(self, sents, sep=None, offset=0):
        """
        Format timestamped words or sentences into a single sequence.
        
        Parameters
        ----------
        sents : list
            List of tuples (start_time, end_time, text).
        sep : str, optional
            Separator for joining texts. Defaults to ASR's separator.
        offset : float, optional
            Time offset to add to timestamps.
            
        Returns
        -------
        tuple
            (start_time, end_time, text) or (None, None, "") if empty.
        """
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b, e, t)


class VACOnlineASRProcessor(OnlineASRProcessor):
    """
    Online ASR processor with Voice Activity Control.
    
    This class wraps OnlineASRProcessor with VAC (Voice Activity Controller),
    detecting speech segments and processing them in real-time.
    
    Parameters
    ----------
    online_chunk_size : float
        Size of online chunks in seconds.
    *a : positional arguments
        Arguments to pass to OnlineASRProcessor.
    **kw : keyword arguments
        Keyword arguments to pass to OnlineASRProcessor.
    """

    def __init__(self, online_chunk_size, *a, **kw):
        """
        Initialize the VACOnlineASRProcessor.
        
        Parameters
        ----------
        online_chunk_size : float
            Size of online chunks in seconds.
        *a : positional arguments
            Arguments to pass to OnlineASRProcessor.
        **kw : keyword arguments
            Keyword arguments to pass to OnlineASRProcessor.
        """
        self.online_chunk_size = online_chunk_size

        self.online = OnlineASRProcessor(*a, **kw)

        # VAC:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        # Use the relative import from the package
        from whisper_streaming.silero_vad_iterator import FixedVADIterator
        self.vac = FixedVADIterator(model)  # we use the default options there: 500ms silence, 100ms padding, etc.  

        self.logfile = self.online.logfile
        self.init()

    def init(self):
        """
        Initialize or reset the processor.
        """
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0

        self.is_currently_final = False

        self.status = None  # or "voice" or "nonvoice"
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def clear_buffer(self):
        """
        Clear the audio buffer and update the offset.
        """
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio):
        """
        Add an audio chunk and process it with VAD.
        
        Parameters
        ----------
        audio : numpy.ndarray
            Audio data as float32 array.
        """
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0] - self.buffer_offset
            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                send_audio = self.audio_buffer[frame:]
                self.online.init(offset=(frame+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[:frame]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            else:
                beg = res["start"] - self.buffer_offset
                end = res["end"] - self.buffer_offset
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[beg:end]
                self.online.init(offset=(beg+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
        else:
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # We keep 1 second because VAD may later find start of voice in it.
                # But we trim it to prevent OOM. 
                self.buffer_offset += max(0, len(self.audio_buffer)-self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]

    def process_iter(self):
        """
        Process the current audio buffer and return transcription.
        
        Returns
        -------
        tuple
            (start_time, end_time, text) or (None, None, "").
        """
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE*self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            return ret
        else:
            print("No online update, only VAD", self.status, file=self.logfile)
            return (None, None, "")

    def finish(self):
        """
        Finalize processing and return any remaining transcription.
        
        Returns
        -------
        tuple
            (start_time, end_time, text) or (None, None, "").
        """
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret


def asr_factory(args, logfile=sys.stderr):
    """
    Create ASR and OnlineASR instances based on command-line arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments including:
        - backend: ASR backend to use
        - model: Model size/name
        - lan: Language code
        - model_cache_dir: Optional cache directory
        - model_dir: Optional model directory
        - vad: Whether to use VAD
        - vac: Whether to use VAC
        - vac_chunk_size: VAC sample size
        - min_chunk_size: Minimum chunk size
        - buffer_trimming: Buffer trimming strategy
        - buffer_trimming_sec: Buffer trimming threshold
        - task: 'transcribe' or 'translate'
    logfile : file, optional
        File for logging, default is sys.stderr.
        
    Returns
    -------
    tuple
        (asr, online) - ASR backend and online processor instances.
    """
    backend = args.backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        asr = OpenaiApiASR(lan=args.lan)
    else:
        if backend == "faster-whisper":
            asr_cls = FasterWhisperASR
        elif backend == "mlx-whisper":
            asr_cls = MLXWhisper
        else:
            asr_cls = WhisperTimestampedASR

        # Only for FasterWhisperASR and WhisperTimestampedASR
        size = args.model
        t = time.time()
        logger.info(f"Loading Whisper {size} model for {args.lan}...")
        asr = asr_cls(modelsize=size, lan=args.lan, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
        e = time.time()
        logger.info(f"Done. It took {round(e-t,2)} seconds.")

    # Apply common configurations
    if getattr(args, 'vad', False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad()

    language = args.lan
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language

    # Create the tokenizer
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None

    # Create the OnlineASRProcessor
    if args.vac:
        online = VACOnlineASRProcessor(args.min_chunk_size, asr, tokenizer, logfile=logfile, buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    else:
        online = OnlineASRProcessor(asr, tokenizer, logfile=logfile, buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))

    return asr, online


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str, help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
    add_shared_args(parser)
    parser.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    parser.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    parser.add_argument('--comp_unaware', action="store_true", default=False, help='Computationally unaware simulation.')
    
    args = parser.parse_args()

    # Reset to store stderr to different file stream, e.g. open(os.devnull,"w")
    logfile = sys.stderr

    if args.offline and args.comp_unaware:
        logger.error("No or one option from --offline and --comp_unaware are available, not both. Exiting.")
        sys.exit(1)

    set_logging(args, logger)

    audio_path = args.audio_path

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path))/SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    asr, online = asr_factory(args, logfile=logfile)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size

    # Load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(audio_path, 0, 1)

    # Warm up the ASR because the very first transcribe takes much more time than the other
    asr.transcribe(a)

    beg = args.start_at
    start = time.time() - beg

    def output_transcript(o, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time() - start
        if o[0] is not None:
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000, o[1]*1000, o[2]), file=logfile, flush=True)
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000, o[1]*1000, o[2]), flush=True)
        else:
            # No text, so no output
            pass

    if args.offline:  # Offline mode processing (for testing/debugging)
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
        except AssertionError as e:
            logger.error(f"Assertion error: {repr(e)}")
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # Computational unaware mode 
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path, beg, end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"Assertion error: {repr(e)}")
                pass
            else:
                output_transcript(o, now=end)

            logger.debug(f"## Last processed {end:.2f}s")

            if end >= duration:
                break
            
            beg = end
            
            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration
    else:  # Online = simultaneous mode
        end = 0
        while True:
            now = time.time() - start
            if now < end + min_chunk:
                time.sleep(min_chunk + end - now)
            end = time.time() - start
            a = load_audio_chunk(audio_path, beg, end)
            beg = end
            online.insert_audio_chunk(a)

            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"Assertion error: {e}")
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            logger.debug(f"## Last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}")

            if end >= duration:
                break
        now = None

    o = online.finish()
    output_transcript(o, now=now)
