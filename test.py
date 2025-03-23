#!/usr/bin/env python3
"""
Example application demonstrating the use of the whisper_streaming package
with a client-server architecture for real-time speech transcription.

This script provides a CLI interface to start either the server, client, or both
in separate processes, with configuration options for each component.
"""

import argparse
import asyncio
import logging
import multiprocessing
import os
import sys
import time
from typing import Dict, Optional

from transcription_server import TranscriptionServer
from transcription_client import TranscriptionClient


class WhisperStreamingApp:
    """
    Main application class for the whisper_streaming example.
    
    This class provides methods to start and manage the transcription server
    and client components, either individually or together.
    
    Parameters
    ----------
    server_config : Dict
        Configuration options for the TranscriptionServer
    client_config : Dict
        Configuration options for the TranscriptionClient
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Attributes
    ----------
    server_config : Dict
        Server configuration
    client_config : Dict
        Client configuration
    logger : logging.Logger
        Logger instance
    server_process : Optional[multiprocessing.Process]
        Server process when running in multi-process mode
    client_process : Optional[multiprocessing.Process]
        Client process when running in multi-process mode
    """
    
    def __init__(
        self,
        server_config: Dict,
        client_config: Dict,
        log_level: str = "INFO"
    ):
        """
        Initialize the WhisperStreamingApp with configuration parameters.
        
        Parameters
        ----------
        server_config : Dict
            Server configuration options
        client_config : Dict
            Client configuration options
        log_level : str, optional
            Logging level, by default "INFO"
        """
        self.server_config = server_config
        self.client_config = client_config
        
        # Initialize logger
        self.logger = logging.getLogger("whisper_streaming_app")
        log_level_value = getattr(logging, log_level)
        self.logger.setLevel(log_level_value)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize process variables
        self.server_process = None
        self.client_process = None
    
    def start_server(self):
        """
        Start the transcription server in the current process.
        
        Returns
        -------
        bool
            True if server started successfully, False otherwise
        """
        try:
            self.logger.info("Starting transcription server")
            
            # Create and start server
            server = TranscriptionServer(
                host=self.server_config.get("host", "127.0.0.1"),
                port=self.server_config.get("port", 43001),
                model_path=self.server_config.get("model_path"),
                model_size=self.server_config.get("model_size", "base.en"),
                language=self.server_config.get("language", "en"),
                log_level=self.server_config.get("log_level", "INFO")
            )
            
            # Run server asynchronously
            loop = asyncio.get_event_loop()
            server_task = loop.create_task(server.start())
            
            # Keep running until interrupted
            try:
                loop.run_until_complete(server_task)
            except KeyboardInterrupt:
                self.logger.info("Server stopped by user")
            finally:
                # Clean up
                pending_tasks = asyncio.all_tasks(loop)
                for task in pending_tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
                loop.run_until_complete(server.stop())
                loop.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            return False
    
    def start_client(self):
        """
        Start the transcription client in the current process.
        
        Returns
        -------
        bool
            True if client started successfully, False otherwise
        """
        try:
            self.logger.info("Starting transcription client")
            
            # Create and start client
            client = TranscriptionClient(
                host=self.client_config.get("host", "127.0.0.1"),
                port=self.client_config.get("port", 43001),
                chunk_size=self.client_config.get("chunk_size", 1024),
                sample_rate=self.client_config.get("sample_rate", 16000),
                log_level=self.client_config.get("log_level", "INFO")
            )
            
            # Run client asynchronously
            loop = asyncio.get_event_loop()
            
            try:
                loop.run_until_complete(client.run())
            except KeyboardInterrupt:
                self.logger.info("Client stopped by user")
            finally:
                # Clean up
                pending_tasks = asyncio.all_tasks(loop)
                for task in pending_tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
                loop.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Error starting client: {e}")
            return False
    
    def _server_process_target(self):
        """
        Target function for the server process.
        """
        try:
            self.start_server()
        except Exception as e:
            print(f"Server process error: {e}")
            sys.exit(1)
    
    def _client_process_target(self):
        """
        Target function for the client process.
        """
        try:
            # Wait a moment to ensure server is ready
            time.sleep(2)
            self.start_client()
        except Exception as e:
            print(f"Client process error: {e}")
            sys.exit(1)
    
    def start_both(self):
        """
        Start both the server and client in separate processes.
        
        Returns
        -------
        bool
            True if both components started successfully, False otherwise
        """
        try:
            self.logger.info("Starting server and client in separate processes")
            
            # Start server process
            self.server_process = multiprocessing.Process(
                target=self._server_process_target,
                name="ServerProcess"
            )
            self.server_process.start()
            
            # Start client process
            self.client_process = multiprocessing.Process(
                target=self._client_process_target,
                name="ClientProcess"
            )
            self.client_process.start()
            
            # Wait for processes to finish
            try:
                while True:
                    if not self.server_process.is_alive() and not self.client_process.is_alive():
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.logger.info("Application stopped by user")
                self.stop()
            
            return True
        except Exception as e:
            self.logger.error(f"Error starting application: {e}")
            self.stop()
            return False
    
    def stop(self):
        """
        Stop all running processes.
        """
        self.logger.info("Stopping application")
        
        # Stop client process
        if self.client_process and self.client_process.is_alive():
            self.logger.info("Terminating client process")
            self.client_process.terminate()
            self.client_process.join(timeout=5)
            if self.client_process.is_alive():
                self.client_process.kill()
            self.client_process = None
        
        # Stop server process
        if self.server_process and self.server_process.is_alive():
            self.logger.info("Terminating server process")
            self.server_process.terminate()
            self.server_process.join(timeout=5)
            if self.server_process.is_alive():
                self.server_process.kill()
            self.server_process = None
        
        self.logger.info("Application stopped")


def main():
    """
    Main function to start the example application from command line.
    """
    parser = argparse.ArgumentParser(description="Whisper Streaming Example Application")
    
    # Main command options
    parser.add_argument("mode", choices=["server", "client", "both"],
                        help="Run mode: server, client, or both")
    
    # Server options
    parser.add_argument("--server-host", type=str, default="127.0.0.1",
                        help="Server hostname or IP")
    parser.add_argument("--server-port", type=int, default=43001,
                        help="Server port")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to local Whisper model")
    parser.add_argument("--model-size", type=str, default="base.en",
                        help="Whisper model size")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code for transcription")
    
    # Client options
    parser.add_argument("--client-host", type=str, default="127.0.0.1",
                        help="Server hostname or IP to connect to")
    parser.add_argument("--client-port", type=int, default=43001,
                        help="Server port to connect to")
    parser.add_argument("--chunk-size", type=int, default=1024,
                        help="Audio chunk size")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Audio sample rate")
    
    # General options
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Configure server
    server_config = {
        "host": args.server_host,
        "port": args.server_port,
        "model_path": args.model_path,
        "model_size": args.model_size,
        "language": args.language,
        "log_level": args.log_level
    }
    
    # Configure client
    client_config = {
        "host": args.client_host,
        "port": args.client_port,
        "chunk_size": args.chunk_size,
        "sample_rate": args.sample_rate,
        "log_level": args.log_level
    }
    
    # Create app
    app = WhisperStreamingApp(
        server_config=server_config,
        client_config=client_config,
        log_level=args.log_level
    )
    
    # Run in selected mode
    if args.mode == "server":
        app.start_server()
    elif args.mode == "client":
        app.start_client()
    elif args.mode == "both":
        app.start_both()


if __name__ == "__main__":
    # For Windows multiprocessing support
    multiprocessing.freeze_support()
    main()
