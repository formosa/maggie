"""
Maggie AI Telemetry and Logging System

Provides advanced logging, performance tracking, and optional 
anonymized usage statistics for continuous improvement.
"""

import os
import json
import uuid
import platform
import logging
import threading
import traceback
import sys  # Added missing import
from datetime import datetime
from typing import Dict, Any, Optional
import psutil
import requests
import torch

class MaggieTelemetryManager:
    """
    Advanced telemetry and logging management for Maggie AI Assistant.

    Manages system logging, performance tracking, and optional 
    anonymized usage statistics with robust privacy controls.

    Attributes
    ----------
    _instance_id : str
        Unique identifier for the current Maggie AI installation
    _config : Dict[str, Any]
        Telemetry configuration settings
    _logger : logging.Logger
        Primary logging interface for the application
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Maggie Telemetry Manager.

        Parameters
        ----------
        config_path : Optional[str], optional
            Path to the telemetry configuration file
        """
        self._instance_id = str(uuid.uuid4())
        self._config = self._load_telemetry_config(config_path)
        self._logger = self._configure_logging()

    def _load_telemetry_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load telemetry configuration with sensible defaults.

        Parameters
        ----------
        config_path : Optional[str], optional
            Path to the configuration file, by default None

        Returns
        -------
        Dict[str, Any]
            Telemetry configuration dictionary with default values overridden by
            user configuration when available
        """
        default_config = {
            "logging": {
                "level": "INFO",
                "filepath": "logs/maggie_telemetry.log"
            },
            "telemetry": {
                "opt_in": False,
                "endpoint": "https://telemetry.maggieai.com/report"
            }
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    return {**default_config, **user_config}
            except (IOError, json.JSONDecodeError):
                self._logger.warning("Invalid telemetry config. Using defaults.")

        return default_config

    def _configure_logging(self) -> logging.Logger:
        """
        Configure comprehensive logging mechanism.

        Returns
        -------
        logging.Logger
            Configured logging instance
        """
        log_config = self._config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        log_filepath = log_config.get('filepath', 'logs/maggie_telemetry.log')

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filepath),
                logging.StreamHandler()
            ]
        )

        return logging.getLogger('MaggieTelemetry')

    def capture_system_snapshot(self) -> Dict[str, Any]:
        """
        Capture comprehensive system configuration snapshot.

        Returns
        -------
        Dict[str, Any]
            Detailed system configuration information
        """
        system_info = {
            "instance_id": self._instance_id,
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "os": platform.system(),
                "release": platform.release(),
                "architecture": platform.machine(),
                "python_version": platform.python_version()
            },
            "hardware": {
                "cpu": {
                    "name": platform.processor(),
                    "physical_cores": os.cpu_count()
                },
                "memory": {
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
                }
            }
        }

        # Extend with GPU information if available
        try:
            system_info['hardware']['gpu'] = {
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
            }
        except ImportError:
            system_info['hardware']['gpu'] = {"cuda_available": False}

        return system_info

    def log_installation_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log significant installation or runtime events.

        Parameters
        ----------
        event_type : str
            Type of event being logged
        details : Dict[str, Any]
            Additional event-specific details
        """
        event_log = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **details
        }
        self._logger.info(f"Event Logged: {json.dumps(event_log)}")

    def submit_anonymous_telemetry(self, event_data: Dict[str, Any]) -> bool:
        """
        Submit anonymized telemetry data if user has opted in.

        Parameters
        ----------
        event_data : Dict[str, Any]
            Telemetry event data to be submitted

        Returns
        -------
        bool
            Submission success status
        """
        if not self._config['telemetry']['opt_in']:
            return False

        try:
            response = requests.post(
                self._config['telemetry']['endpoint'],
                json=event_data,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            self._logger.error(f"Telemetry submission failed: {e}")
            return False

def global_exception_handler(exc_type, exc_value, exc_traceback):
    """
    Global exception handler for unhandled exceptions.

    Logs critical errors and potentially submits anonymized error reports.
    """
    error_details = {
        "type": str(exc_type.__name__),
        "message": str(exc_value),
        "traceback": ''.join(traceback.format_tb(exc_traceback))
    }
    
    telemetry_manager = MaggieTelemetryManager()
    telemetry_manager._logger.critical(f"Unhandled Exception: {json.dumps(error_details)}")
    
    # Optionally submit error telemetry
    telemetry_manager.submit_anonymous_telemetry({
        "event_type": "unhandled_exception",
        **error_details
    })

# Configure global exception handling
sys.excepthook = global_exception_handler