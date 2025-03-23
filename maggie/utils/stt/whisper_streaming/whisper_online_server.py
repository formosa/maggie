#!/usr/bin/env python3
"""
Whisper Online Server Module

This module provides a server implementation for the whisper_streaming package,
enabling real-time audio transcription over a network connection.

The server listens for incoming audio data from clients, processes it using
the specified ASR backend, and returns transcription results.
"""

import sys
import argparse
import os
import logging
import numpy as np
import socket
import io
import soundfile

# Import from the whisper_streaming package
from whisper_streaming.whisper_online import (
    load_audio_chunk, asr_factory, set_logging, add_shared_args
)
from whisper_streaming.line_packet import (
    send_one_line, receive_one_line, receive_lines, PACKET_SIZE
)

# Configure module logger
logger = logging.getLogger(__name__)


class Connection:
    """
    Wrapper for socket connections in the transcription server.
    
    This class handles sending and receiving text lines over sockets,
    simplifying communication in the server context.
    
    Parameters
    ----------
    conn : socket.socket
        Socket connection object.
        
    Attributes
    ----------
    conn : socket.socket
        The socket connection
    last_line : str
        The last line sent, to avoid duplicates
    PACKET_SIZE : int
        Size of the packet buffer (32000*5*60)
    """

    PACKET_SIZE = 32000 * 5 * 60  # 5 minutes (was: 65536)

    def __init__(self, conn):
        """
        Initialize the connection wrapper.
        
        Parameters
        ----------
        conn : socket.socket
            Socket connection object.
        """
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        """
        Send a line of text, avoiding duplicates.
        
        Parameters
        ----------
        line : str
            Line of text to send.
            
        Notes
        -----
        If the line is identical to the last sent line, it won't be sent again.
        This helps prevent issues in online-text-flow-events.
        """
        if line == self.last_line:
            return
        send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        """
        Receive multiple lines of text.
        
        Returns
        -------
        list or None
            List of received lines or None if connection closed.
        """
        in_line = receive_lines(self.conn)
        return in_line

    def non_blocking_receive_audio(self):
        """
        Receive audio data without blocking.
        
        Returns
        -------
        bytes or None
            Received audio data or None if error or no data.
        """
        try:
            r = self.conn.recv(self.PACKET_SIZE)
            return r
        except ConnectionResetError:
            return None


class ServerProcessor:
    """
    Processor for audio chunks in server context.
    
    This class handles receiving audio, processing it, and sending
    transcription results to clients.
    
    Parameters
    ----------
    c : Connection
        Connection wrapper for socket communication.
    online_asr_proc : OnlineASRProcessor
        Processor for online audio transcription.
    min_chunk : float
        Minimum chunk size in seconds.
        
    Attributes
    ----------
    connection : Connection
        Connection wrapper
    online_asr_proc : OnlineASRProcessor
        Transcription processor
    min_chunk : float
        Minimum audio chunk size in seconds
    last_end : float or None
        End timestamp of the last processed segment
    is_first : bool
        Flag indicating if this is the first chunk
    """

    def __init__(self, c, online_asr_proc, min_chunk):
        """
        Initialize the server processor.
        
        Parameters
        ----------
        c : Connection
            Connection wrapper for socket communication.
        online_asr_proc : OnlineASRProcessor
            Processor for online audio transcription.
        min_chunk : float
            Minimum chunk size in seconds.
        """
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

        self.last_end = None

        self.is_first = True

    def receive_audio_chunk(self):
        """
        Receive and process audio chunks from the client.
        
        Returns
        -------
        numpy.ndarray or None
            Audio data as float32 array or None if no data.
            
        Notes
        -----
        Blocks operation if less than min_chunk seconds is available.
        Unblocks if connection is closed or a chunk is available.
        """
        # Receive all audio that is available by this time
        # Blocks operation if less than self.min_chunk seconds is available
        # Unblocks if connection is closed or a chunk is available
        out = []
        minlimit = self.min_chunk * 16000  # SAMPLING_RATE
        while sum(len(x) for x in out) < minlimit:
            raw_bytes = self.connection.non_blocking_receive_audio()
            if not raw_bytes:
                break
            # Convert raw bytes to audio samples
            sf_file = soundfile.SoundFile(
                io.BytesIO(raw_bytes), 
                channels=1,
                endian="LITTLE",
                samplerate=16000,
                subtype="PCM_16",
                format="RAW"
            )
            audio, _ = librosa.load(sf_file, sr=16000, dtype=np.float32)
            out.append(audio)
            
        if not out:
            return None
            
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
            
        self.is_first = False
        return np.concatenate(out)

    def format_output_transcript(self, o):
        """
        Format transcription output for sending to client.
        
        Parameters
        ----------
        o : tuple
            (start_time, end_time, text) from ASR processor.
            
        Returns
        -------
        str or None
            Formatted output string or None.
            
        Notes
        -----
        Output format is: "beg_time end_time text"
        where beg_time and end_time are in milliseconds.
        
        Succeeding intervals are adjusted to not overlap because ELITR protocol
        requires it. Therefore, beg is max of previous end and current beg.
        """
        # Output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol requires it.
        # Therefore, beg is max of previous end and current beg outputed by Whisper.

        if o[0] is not None:
            beg, end = o[0] * 1000, o[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            print("%1.0f %1.0f %s" % (beg, end, o[2]), flush=True, file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg, end, o[2])
        else:
            logger.debug("No text in this segment")
            return None

    def send_result(self, o):
        """
        Send transcription result to the client.
        
        Parameters
        ----------
        o : tuple
            (start_time, end_time, text) from ASR processor.
        """
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)

    def process(self):
        """
        Handle one client connection lifecycle.
        
        This method continuously receives audio chunks, processes them,
        and sends back transcription results until the connection is closed.
        """
        # Initialize the ASR processor
        self.online_asr_proc.init()
        
        # Main processing loop
        while True:
            # Receive audio chunk
            a = self.receive_audio_chunk()
            if a is None:
                break
                
            # Process audio
            self.online_asr_proc.insert_audio_chunk(a)
            o = self.online_asr_proc.process_iter()
            
            # Send results
            try:
                self.send_result(o)
            except BrokenPipeError:
                logger.info("Broken pipe -- connection closed?")
                break

        # Optionally process any remaining audio (currently commented out)
        # o = self.online_asr_proc.finish()
        # self.send_result(o)


def main():
    """
    Main function to run the transcription server.
    
    Parses command line arguments, initializes the ASR model and server,
    and handles client connections.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # Server options
    parser.add_argument("--host", type=str, default='localhost',
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=43007,
                        help="Port to listen on")
    parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
                        help="Path to a speech audio wav file to warm up Whisper")

    # Add shared whisper_online arguments
    add_shared_args(parser)
    
    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    set_logging(args, logger, other="")

    # Initialize ASR model
    size = args.model
    language = args.lan
    asr, online = asr_factory(args)
    min_chunk = args.min_chunk_size

    # Warm up the ASR to reduce latency on first request
    msg = "Whisper is not warmed up. The first chunk processing may take longer."
    if args.warmup_file:
        if os.path.isfile(args.warmup_file):
            # Import here to avoid circular import
            from whisper_streaming.whisper_online import load_audio_chunk
            a = load_audio_chunk(args.warmup_file, 0, 1)
            asr.transcribe(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.critical("The warm up file is not available. " + msg)
            sys.exit(1)
    else:
        logger.warning(msg)

    # Start server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.host, args.port))
        s.listen(1)
        logger.info(f'Listening on {args.host}:{args.port}')
        
        # Handle connections
        while True:
            conn, addr = s.accept()
            logger.info(f'Connected to client on {addr}')
            connection = Connection(conn)
            proc = ServerProcessor(connection, online, args.min_chunk_size)
            proc.process()
            conn.close()
            logger.info('Connection to client closed')
            
    logger.info('Connection closed, terminating.')


# Module import guard
if __name__ == "__main__":
    # Additional imports needed only when running as script
    import librosa
    main()
