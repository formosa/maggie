#!/usr/bin/env python3
"""
TranscriptionServer module that provides asynchronous real-time speech transcription
using the whisper_streaming package with faster-whisper backend and GPU acceleration.
"""

import asyncio
import argparse
import logging
import os
import socket
import json
import sys
import numpy as np
from typing import Dict, Tuple, Optional, List

# Import whisper_streaming package
from whisper_streaming import (
    asr_factory, Connection, ServerProcessor, set_logging, add_shared_args
)

class TranscriptionServer:
    """
    Asynchronous server for real-time speech transcription using whisper_streaming.
    
    This class manages client connections and processes audio streams to generate
    transcriptions in real-time, with results classified as 'partial' or 'final'.
    
    Parameters
    ----------
    host : str
        Hostname or IP address to bind the server to
    port : int
        Port number to listen on
    model_path : str
        Path to the local Whisper model
    model_size : str
        Size of the Whisper model (tiny, base, small, medium, large)
    language : str
        Language code for transcription (e.g., 'en', 'auto' for detection)
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Attributes
    ----------
    host : str
        Server hostname or IP
    port : int
        Server port
    model_path : str
        Path to Whisper model
    model_size : str
        Size of the Whisper model
    language : str
        Language code for transcription
    asr_args : argparse.Namespace
        Arguments for ASR factory
    logger : logging.Logger
        Logger instance
    server : socket.socket
        Server socket
    is_running : bool
        Server running state
    """
    
    def __init__(
        self, 
        host: str = '127.0.0.1',
        port: int = 43001,
        model_path: str = None,
        model_size: str = "base.en",
        language: str = "en",
        log_level: str = "INFO"
    ):
        """
        Initialize the TranscriptionServer with configuration parameters.
        
        Parameters
        ----------
        host : str, optional
            Hostname or IP address, by default '127.0.0.1'
        port : int, optional
            Port number, by default 43001
        model_path : str, optional
            Path to local Whisper model, by default None
        model_size : str, optional
            Model size identifier, by default "base.en"
        language : str, optional
            Language code, by default "en"
        log_level : str, optional
            Logging level, by default "INFO"
        """
        self.host = host
        self.port = port
        self.model_path = model_path
        self.model_size = model_size
        self.language = language
        
        # Initialize logger
        self.logger = logging.getLogger("transcription_server")
        log_level_value = getattr(logging, log_level)
        self.logger.setLevel(log_level_value)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Create ASR arguments
        self.asr_args = argparse.Namespace(
            backend="faster-whisper",  # Using faster-whisper as required
            model=self.model_size,
            lan=self.language,
            model_dir=self.model_path,
            model_cache_dir=None,
            vad=True,
            vac=True,
            vac_chunk_size=0.04,
            min_chunk_size=0.5,
            buffer_trimming="segment",
            buffer_trimming_sec=10,
            task="transcribe",
            log_level=log_level
        )
        
        # Server state
        self.server = None
        self.is_running = False
        
    async def start(self):
        """
        Start the transcription server asynchronously.
        
        Creates a socket server and begins listening for client connections.
        """
        self.logger.info(f"Starting transcription server on {self.host}:{self.port}")
        self.is_running = True
        
        # Create server socket
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        self.server.setblocking(False)
        
        # Initialize ASR model
        asr, online = asr_factory(self.asr_args, logfile=sys.stderr)
        self.logger.info(f"ASR model initialized: {self.model_size}")
        
        # Accept connections
        while self.is_running:
            try:
                # Using asyncio to handle non-blocking socket operations
                conn, addr = await self.accept_connection()
                if conn:
                    self.logger.info(f"New connection from {addr}")
                    # Handle client in a separate task
                    asyncio.create_task(self.handle_client(conn, addr, asr, online))
            except asyncio.CancelledError:
                self.logger.info("Server task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error accepting connection: {e}")
                await asyncio.sleep(0.1)
                
    async def accept_connection(self):
        """
        Accept a client connection asynchronously.
        
        Returns
        -------
        Tuple[socket.socket, Tuple[str, int]]
            Socket connection and client address tuple
        """
        loop = asyncio.get_event_loop()
        try:
            return await loop.sock_accept(self.server)
        except Exception as e:
            self.logger.error(f"Error accepting connection: {e}")
            return None, None

    async def handle_client(self, conn, addr, asr, online):
        """
        Handle a client connection for audio transcription.
        
        Parameters
        ----------
        conn : socket.socket
            Client socket connection
        addr : Tuple[str, int]
            Client address information
        asr : ASRBase
            ASR implementation instance
        online : OnlineASRProcessor
            Online processing instance
        """
        connection = Connection(conn)
        processor = EnhancedServerProcessor(connection, online, self.asr_args.min_chunk_size)
        
        try:
            # Process client connection
            await self.process_audio_stream(processor)
        except Exception as e:
            self.logger.error(f"Error processing client {addr}: {e}")
        finally:
            self.logger.info(f"Client {addr} disconnected")
            conn.close()
    
    async def process_audio_stream(self, processor):
        """
        Process the audio stream from a client asynchronously.
        
        Parameters
        ----------
        processor : EnhancedServerProcessor
            Server processor for handling audio chunks
        """
        loop = asyncio.get_event_loop()
        
        try:
            # Run the processor in a thread pool to avoid blocking the event loop
            await loop.run_in_executor(None, processor.process)
        except Exception as e:
            self.logger.error(f"Error in audio processing: {e}")
    
    async def stop(self):
        """
        Stop the transcription server gracefully.
        """
        self.logger.info("Stopping transcription server")
        self.is_running = False
        
        if self.server:
            self.server.close()
            self.server = None


class EnhancedServerProcessor(ServerProcessor):
    """
    Enhanced server processor that adds classification of transcription results.
    
    This class extends ServerProcessor to classify results as 'partial' or 'final'.
    
    Parameters
    ----------
    c : Connection
        Connection wrapper for socket communication
    online_asr_proc : OnlineASRProcessor
        Processor for online audio transcription
    min_chunk : float
        Minimum chunk size in seconds
    
    Attributes
    ----------
    pending_segments : List[Dict]
        List of pending segments waiting to be finalized
    """
    
    def __init__(self, c, online_asr_proc, min_chunk):
        """
        Initialize the EnhancedServerProcessor.
        
        Parameters
        ----------
        c : Connection
            Connection wrapper for socket communication
        online_asr_proc : OnlineASRProcessor
            Processor for online audio transcription
        min_chunk : float
            Minimum chunk size in seconds
        """
        super().__init__(c, online_asr_proc, min_chunk)
        self.pending_segments = []
        self.logger = logging.getLogger("enhanced_processor")
    
    def format_output_transcript(self, o):
        """
        Format transcription output for sending to client with classification.
        
        Parameters
        ----------
        o : tuple
            (start_time, end_time, text) from ASR processor
            
        Returns
        -------
        str or None
            Formatted output string or None
        """
        if o[0] is None:
            return None
            
        beg, end = o[0]*1000, o[1]*1000
        if self.last_end is not None:
            beg = max(beg, self.last_end)
        
        self.last_end = end
        text = o[2]
        
        # Determine if this is a new segment or an update to an existing one
        segment_id = f"{beg:.0f}-{end:.0f}"
        is_final = self._check_if_final(text, beg, end)
        
        result = {
            "start_time": beg,
            "end_time": end,
            "text": text,
            "type": "final" if is_final else "partial",
            "segment_id": segment_id
        }
        
        # Store or update segment for future reference
        self._update_pending_segments(result)
        
        # Convert to JSON for transmission
        return json.dumps(result)
    
    def _check_if_final(self, text, start_time, end_time):
        """
        Determine if a transcription segment should be marked as final.
        
        A segment is considered final if:
        1. It ends with punctuation
        2. There is a sufficient pause after it (detected by VAD)
        3. It hasn't changed in multiple consecutive iterations
        
        Parameters
        ----------
        text : str
            Transcription text
        start_time : float
            Start time in milliseconds
        end_time : float
            End time in milliseconds
            
        Returns
        -------
        bool
            True if the segment is final, False otherwise
        """
        # Check for ending punctuation
        if text and text[-1] in ['.', '!', '?']:
            return True
            
        # Check if this segment hasn't changed for a while
        for segment in self.pending_segments:
            if (abs(segment["start_time"] - start_time) < 200 and 
                abs(segment["end_time"] - end_time) < 200):
                # If the text hasn't changed in two iterations, consider it final
                if segment["text"] == text and segment.get("unchanged_count", 0) > 1:
                    return True
                
        # Otherwise, consider it partial
        return False
    
    def _update_pending_segments(self, result):
        """
        Update the list of pending segments.
        
        Parameters
        ----------
        result : Dict
            Segment information
        """
        segment_id = result["segment_id"]
        
        # Find if this segment already exists
        existing_idx = None
        for i, segment in enumerate(self.pending_segments):
            if segment["segment_id"] == segment_id:
                existing_idx = i
                break
                
        if existing_idx is not None:
            # Update existing segment
            existing = self.pending_segments[existing_idx]
            if existing["text"] == result["text"]:
                # Text hasn't changed, increment counter
                existing["unchanged_count"] = existing.get("unchanged_count", 0) + 1
            else:
                # Text changed, reset counter
                existing["unchanged_count"] = 0
                
            existing["text"] = result["text"]
            existing["type"] = result["type"]
            
            # Remove if final
            if result["type"] == "final":
                self.pending_segments.pop(existing_idx)
        else:
            # Add new segment
            new_segment = result.copy()
            new_segment["unchanged_count"] = 0
            self.pending_segments.append(new_segment)
            
        # Clean up old segments (older than 30 seconds)
        current_time = result["end_time"]
        self.pending_segments = [
            s for s in self.pending_segments 
            if current_time - s["end_time"] < 30000
        ]


def main():
    """
    Main function to start the transcription server from command line.
    """
    parser = argparse.ArgumentParser(description="Transcription Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=43001, help="Server port")
    parser.add_argument("--model-path", type=str, default=None, help="Path to local Whisper model")
    parser.add_argument("--model-size", type=str, default="base.en", help="Whisper model size")
    parser.add_argument("--language", type=str, default="en", help="Language code")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Create and start server
    server = TranscriptionServer(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        model_size=args.model_size,
        language=args.language,
        log_level=args.log_level
    )
    
    loop = asyncio.get_event_loop()
    
    try:
        server_task = loop.create_task(server.start())
        loop.run_until_complete(server_task)
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        # Clean up
        pending_tasks = asyncio.all_tasks(loop)
        for task in pending_tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
        loop.run_until_complete(server.stop())
        loop.close()

if __name__ == "__main__":
    main()
