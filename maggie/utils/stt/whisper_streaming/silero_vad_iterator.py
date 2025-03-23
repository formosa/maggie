"""
Voice Activity Detection (VAD) iterators for audio stream processing.

This module provides iterators for voice activity detection using the Silero VAD model.
It includes a base VADIterator class and an extended FixedVADIterator that supports
variable-length inputs, making it suitable for real-time audio processing.

The implementation is based on silero-vad with MIT license:
https://github.com/snakers4/silero-vad
"""

import torch
import numpy as np


class VADIterator:
    """
    Voice Activity Detection iterator for streaming audio processing.
    
    This class provides methods to process audio chunks and detect speech segments
    in an incremental, streaming fashion. It maintains state between calls to
    enable continuous processing of audio streams.
    
    Parameters
    ----------
    model : torch.nn.Module
        Preloaded Silero VAD model.
    threshold : float, optional
        Speech probability threshold. Values above this are considered speech.
        Default is 0.5.
    sampling_rate : int, optional
        Audio sampling rate, must be 8000 or 16000 Hz. Default is 16000.
    min_silence_duration_ms : int, optional
        Minimum silence duration in milliseconds before separating speech chunks.
        Default is 500ms.
    speech_pad_ms : int, optional
        Padding added to each side of detected speech segments in milliseconds.
        Default is 100ms.
        
    Notes
    -----
    This class implements a stateful iterator pattern. Each call to __call__
    processes a single audio chunk and returns speech segment boundaries when detected.
    """

    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 500,
                 speech_pad_ms: int = 100):
        """
        Initialize the VAD iterator with model and configuration parameters.
        
        Parameters
        ----------
        model : torch.nn.Module
            Preloaded Silero VAD model.
        threshold : float, optional
            Speech probability threshold (0.0 to 1.0). Default is 0.5.
        sampling_rate : int, optional
            Audio sampling rate in Hz. Default is 16000.
        min_silence_duration_ms : int, optional
            Minimum silence duration before separating speech chunks, in milliseconds.
            Default is 500ms.
        speech_pad_ms : int, optional
            Additional padding for speech segments, in milliseconds. Default is 100ms.
            
        Raises
        ------
        ValueError
            If sampling_rate is not 8000 or 16000 Hz.
        """
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        """
        Reset the internal state of the VAD iterator.
        
        This method should be called when starting a new audio stream or
        when the continuity of the audio stream is broken.
        """
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False):
        """
        Process an audio chunk and detect speech segments.
        
        Parameters
        ----------
        x : torch.Tensor or numpy.ndarray
            Audio chunk to process. Should be a 1D array or tensor of 
            audio samples.
        return_seconds : bool, optional
            Whether to return timestamps in seconds instead of samples.
            Default is False.
            
        Returns
        -------
        dict or None
            Dictionary with 'start' key when speech begins, 
            'end' key when speech ends, or None if no speech boundary is detected.
            Timestamps are in samples or seconds based on return_seconds parameter.
            
        Notes
        -----
        This method maintains state between calls to enable continuous 
        processing of audio streams.
        """
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - self.speech_pad_samples
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}

        return None


class FixedVADIterator(VADIterator):
    """
    Enhanced VAD iterator supporting variable-length audio inputs.
    
    This class extends the base VADIterator to handle audio chunks of any length,
    not just the 512-sample chunks required by the standard Silero VAD implementation.
    It buffers audio internally and processes it in appropriate sized chunks.
    
    Parameters
    ----------
    model : torch.nn.Module
        Preloaded Silero VAD model.
    threshold : float, optional
        Speech probability threshold. Default is 0.5.
    sampling_rate : int, optional
        Audio sampling rate in Hz. Default is 16000.
    min_silence_duration_ms : int, optional
        Minimum silence duration in milliseconds. Default is 500ms.
    speech_pad_ms : int, optional
        Padding added to speech segments in milliseconds. Default is 100ms.
        
    Notes
    -----
    If multiple voice segments are detected in a single audio chunk,
    only the start of the first segment and the end (or continuation) of the
    last segment will be reported.
    """

    def reset_states(self):
        """
        Reset the internal state of the VAD iterator.
        
        In addition to the base class reset, this also clears the audio buffer.
        """
        super().reset_states()
        self.buffer = np.array([], dtype=np.float32)

    def __call__(self, x, return_seconds=False):
        """
        Process audio of any length for voice activity detection.
        
        This method buffers audio and processes it in 512-sample chunks to work
        with the Silero VAD model. It handles audio of any length, solving the
        limitation of the original VADIterator which requires exactly 512 samples.
        
        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor
            Audio chunk of any length to process.
        return_seconds : bool, optional
            Whether to return timestamps in seconds instead of samples.
            Default is False.
            
        Returns
        -------
        dict or None
            Dictionary with speech segment boundaries or None if no 
            boundary is detected.
            
        Notes
        -----
        When processing long audio and multiple voiced segments are detected,
        this method returns the start of the first segment and the end 
        (or middle, which means no end) of the last segment.
        """
        self.buffer = np.append(self.buffer, x) 
        ret = None
        while len(self.buffer) >= 512:
            r = super().__call__(self.buffer[:512], return_seconds=return_seconds)
            self.buffer = self.buffer[512:]
            if ret is None:
                ret = r
            elif r is not None:
                if 'end' in r:
                    ret['end'] = r['end']  # the latter end
                if 'start' in r and 'end' in ret:  # there is an earlier start.
                    # Remove end, merging this segment with the previous one.
                    del ret['end']
        return ret if ret != {} else None


if __name__ == "__main__":
    # Test/demonstrate the need for FixedVADIterator
    import torch
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    vac = FixedVADIterator(model)
    # This works with both VADIterator and FixedVADIterator
    audio_buffer = np.array([0]*(512), dtype=np.float32)
    vac(audio_buffer)

    # This would crash with VADIterator but works with FixedVADIterator
    # Error message would be: ops.prim.RaiseException("Input audio chunk is too short", "builtins.ValueError")
    audio_buffer = np.array([0]*(512-1), dtype=np.float32)
    vac(audio_buffer)
