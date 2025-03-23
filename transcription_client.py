#!/usr/bin/env python3
"""
TranscriptionClient module for capturing audio from a microphone and sending it
to a TranscriptionServer for real-time speech-to-text processing.
"""

import asyncio
import argparse
import json
import logging
import socket
import sys
import time
import threading
import wave
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pyaudio
from whisper_streaming import receive_one_line, send_one_line

class TranscriptionClient:
    """
    Client for capturing audio from a microphone and streaming it to a 
    transcription server for real-time processing.
    
    
    Parameters
    ----------
    host : str
        Hostname or IP of the transcription server
    port : int
        Port of the transcription server
    chunk_size : int
        Number of audio frames per buffer
    sample_rate : int
        Audio sampling rate in Hz
    format_type : int
        PyAudio format type (e.g., pyaudio.paInt16)
    channels : int
        Number of audio channels (1 for mono, 2 for stereo)
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Attributes
    ----------
    host : str
        Server hostname or IP
    port : int
        Server port
    chunk_size : int
        Audio chunk size
    sample_rate : int
        Audio sampling rate
    format_type : int
        PyAudio format
    channels : int
        Number of audio channels
    logger : logging.Logger
        Logger instance
    pyaudio : pyaudio.PyAudio
        PyAudio instance
    stream : pyaudio.Stream
        Audio input stream
    socket : socket.socket
        Client socket for server connection
    is_running : bool
        Client running state
    is_listening : bool
        Microphone listening state
    transcription_buffer : List[Dict]
        Buffer of received transcriptions
    """
    
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 43001,
        chunk_size: int = 1024,
        sample_rate: int = 16000,
        format_type: int = pyaudio.paInt16,
        channels: int = 1,
        log_level: str = "INFO"
    ):
        """
        Initialize the TranscriptionClient with configuration parameters.
        
        Parameters
        ----------
        host : str, optional
            Server hostname or IP, by default '127.0.0.1'
        port : int, optional
            Server port, by default 43001
        chunk_size : int, optional
            Audio chunk size, by default 1024
        sample_rate : int, optional
            Audio sample rate, by default 16000
        format_type : int, optional
            PyAudio format, by default pyaudio.paInt16
        channels : int, optional
            Number of audio channels, by default 1
        log_level : str, optional
            Logging level, by default "INFO"
        """
        self.host = host
        self.port = port
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.format_type = format_type
        self.channels = channels
        
        # Initialize logger
        self.logger = logging.getLogger("transcription_client")
        log_level_value = getattr(logging, log_level)
        self.logger.setLevel(log_level_value)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # Socket connection
        self.socket = None
        
        # State flags
        self.is_running = False
        self.is_listening = False
        
        # Transcription buffer
        self.transcription_buffer = []
        
    async def connect(self):
        """
        Connect to the transcription server asynchronously.
        
        Returns
        -------
        bool
            True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to server at {self.host}:{self.port}")
            
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.is_running = True
            
            self.logger.info("Connected to transcription server")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    async def start_microphone(self):
        """
        Start capturing audio from the microphone and send it to the server.
        
        Returns
        -------
        bool
            True if microphone started successfully, False otherwise
        """
        if not self.socket:
            self.logger.error("Cannot start microphone: Not connected to server")
            return False
            
        try:
            self.logger.info("Starting microphone capture")
            
            # Open audio stream
            self.stream = self.pyaudio.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_listening = True
            
            # Start audio capture in a separate thread
            audio_thread = threading.Thread(target=self._capture_and_send_audio)
            audio_thread.daemon = True
            audio_thread.start()
            
            self.logger.info("Microphone capture started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start microphone: {e}")
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            return False
    
    def _capture_and_send_audio(self):
        """
        Continuously capture audio from microphone and send to server.
        
        This method runs in a separate thread.
        """
        while self.is_listening and self.is_running:
            try:
                # Read audio data
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Convert to raw bytes and send
                if self.socket:
                    self.socket.sendall(audio_data)
                    
            except Exception as e:
                self.logger.error(f"Error in audio capture: {e}")
                if not self.is_running:
                    break
                
        self.logger.info("Audio capture stopped")
    
    async def receive_transcriptions(self):
        """
        Continuously receive and process transcriptions from the server.
        """
        if not self.socket:
            self.logger.error("Cannot receive transcriptions: Not connected to server")
            return
            
        self.logger.info("Starting transcription receiver")
        
        try:
            while self.is_running:
                # Set socket to non-blocking mode for asyncio compatibility
                self.socket.setblocking(False)
                
                try:
                    # Use asyncio to handle non-blocking socket reads
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(None, self._receive_data)
                    
                    if data:
                        # Process received transcription
                        self._process_transcription(data)
                    else:
                        # No data available, short delay
                        await asyncio.sleep(0.01)
                        
                except Exception as e:
                    if "Resource temporarily unavailable" not in str(e):
                        self.logger.error(f"Error receiving transcription: {e}")
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"Transcription receiver error: {e}")
            
        self.logger.info("Transcription receiver stopped")
    
    def _receive_data(self):
        """
        Receive a data packet from the server.
        
        Returns
        -------
        Optional[str]
            Received data or None if no data available
        """
        try:
            # Try to receive data
            result = receive_one_line(self.socket)
            return result
        except BlockingIOError:
            # No data available
            return None
        except Exception as e:
            if self.is_running:
                self.logger.error(f"Error in _receive_data: {e}")
            return None
    
    def _process_transcription(self, data: str):
        """
        Process received transcription data.
        
        Parameters
        ----------
        data : str
            Received transcription data
        """
        try:
            # Parse JSON data
            result = json.loads(data)
            
            # Extract fields
            transcription_type = result.get("type", "partial")
            text = result.get("text", "")
            start_time = result.get("start_time", 0) / 1000.0  # Convert ms to seconds
            end_time = result.get("end_time", 0) / 1000.0
            
            # Print to terminal with appropriate prefix
            prefix = f"{transcription_type}: "
            print(f"{prefix}{text}")
            
            # Store in buffer for later reference
            self._update_transcription_buffer(result)
            
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON data received: {data}")
        except Exception as e:
            self.logger.error(f"Error processing transcription: {e}")
    
    def _update_transcription_buffer(self, result: Dict):
        """
        Update the transcription buffer with new results.
        
        Parameters
        ----------
        result : Dict
            Transcription result
        """
        # Add to buffer and maintain reasonable size
        self.transcription_buffer.append(result)
        
        # Keep only the last 100 transcriptions to avoid memory issues
        if len(self.transcription_buffer) > 100:
            self.transcription_buffer = self.transcription_buffer[-100:]
    
    async def stop_microphone(self):
        """
        Stop microphone capture.
        """
        self.logger.info("Stopping microphone capture")
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        self.logger.info("Microphone capture stopped")
    
    async def disconnect(self):
        """
        Disconnect from the transcription server.
        """
        self.logger.info("Disconnecting from server")
        self.is_running = False
        
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
            
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None
            
        self.logger.info("Disconnected from server")
    
    async def run(self):
        """
        Run the transcription client, connecting to server and processing audio.
        """
        # Connect to server
        if not await self.connect():
            return False
            
        try:
            # Start microphone
            if not await self.start_microphone():
                await self.disconnect()
                return False
                
            # Start receiving transcriptions
            receive_task = asyncio.create_task(self.receive_transcriptions())
            
            # Wait for user to stop
            print("Transcription client is running. Press Ctrl+C to stop.")
            while self.is_running:
                await asyncio.sleep(0.1)
                
            # Wait for tasks to complete
            await receive_task
                
        except asyncio.CancelledError:
            self.logger.info("Client task cancelled")
        except Exception as e:
            self.logger.error(f"Error in run: {e}")
        finally:
            # Clean up
            await self.stop_microphone()
            await self.disconnect()
            
        return True


def main():
    """
    Main function to start the transcription client from command line.
    """
    parser = argparse.ArgumentParser(description="Transcription Client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=43001, help="Server port")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Audio chunk size")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Create and run client
    client = TranscriptionClient(
        host=args.host,
        port=args.port,
        chunk_size=args.chunk_size,
        sample_rate=args.sample_rate,
        log_level=args.log_level
    )
    
    loop = asyncio.get_event_loop()
    
    try:
        loop.run_until_complete(client.run())
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    finally:
        # Clean up
        pending_tasks = asyncio.all_tasks(loop)
        for task in pending_tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
        loop.close()

if __name__ == "__main__":
    main()
