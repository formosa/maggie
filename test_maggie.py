<<<<<<< HEAD
"""
Maggie AI Assistant - Test Script
==============================
Tests critical components of the Maggie AI Assistant.
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional

def run_test(test_name: str, test_func, *args, **kwargs) -> Tuple[bool, str]:
    """
    Run a test function and return the result.
    
    Parameters
    ----------
    test_name : str
        Name of the test
    test_func : callable
        Test function to run
    *args, **kwargs
        Arguments to pass to the test function
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    print(f"Running test: {test_name}...")
    try:
        start_time = time.time()
        result = test_func(*args, **kwargs)
        elapsed = time.time() - start_time
        if isinstance(result, tuple) and len(result) >= 2:
            success, message = result[0], result[1]
        else:
            success, message = result, "Test completed successfully"
        
        status = "PASSED" if success else "FAILED"
        print(f"{status}: {test_name} ({elapsed:.2f}s) - {message}")
        return success, message
    except Exception as e:
        print(f"FAILED: {test_name} - Exception: {str(e)}")
        return False, str(e)

def test_config_loading() -> Tuple[bool, str]:
    """
    Test loading configuration from file.
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    try:
        from utils.config import Config
        
        config = Config().load()
        if not config:
            return False, "Failed to load configuration"
        
        # Check for required sections
        required_sections = ["wake_word", "speech", "llm"]
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            return False, f"Missing required sections: {', '.join(missing_sections)}"
        
        return True, "Configuration loaded successfully"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def test_event_bus() -> Tuple[bool, str]:
    """
    Test the event bus functionality.
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    try:
        from maggie import EventBus
        
        event_bus = EventBus()
        received_events = []
        
        def test_handler(data):
            received_events.append(data)
        
        # Subscribe to test event
        event_bus.subscribe("test_event", test_handler)
        
        # Start event bus
        event_bus.start()
        
        # Publish test event
        event_bus.publish("test_event", "test_data")
        
        # Wait for event processing
        time.sleep(0.5)
        
        # Stop event bus
        event_bus.stop()
        
        if len(received_events) == 1 and received_events[0] == "test_data":
            return True, "Event bus working correctly"
        else:
            return False, f"Event not received correctly, got: {received_events}"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def test_gui_initialization() -> Tuple[bool, str]:
    """
    Test GUI initialization.
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    try:
        from PyQt6.QtWidgets import QApplication
        from utils.gui import MainWindow
        
        # Create QApplication instance
        app = QApplication([])
        
        # Create a mock Maggie object
        class MockMaggie:
            def __init__(self):
                self.utilities = {}
                
            def on_shutdown_clicked(self):
                pass
                
            def on_sleep_clicked(self):
                pass
                
            def process_command(self, utility=None):
                pass
        
        # Create MainWindow
        window = MainWindow(MockMaggie())
        
        # Check if window was created
        if window is not None:
            return True, "GUI initialized successfully"
        else:
            return False, "Failed to initialize GUI"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def test_tts() -> Tuple[bool, str]:
    """
    Test text-to-speech functionality.
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    try:
        from utils.tts import PiperTTS
        
        # Basic TTS configuration
        config = {
            "voice_model": "en_US-kathleen-medium",
            "model_path": "models/tts"
        }
        
        # Check if model files exist
        model_dir = os.path.join(config["model_path"], config["voice_model"])
        onnx_path = os.path.join(model_dir, f"{config['voice_model']}.onnx")
        json_path = os.path.join(model_dir, f"{config['voice_model']}.json")
        
        if not os.path.exists(onnx_path) or not os.path.exists(json_path):
            return False, f"TTS model files not found at {model_dir}"
        
        # Create TTS instance
        tts = PiperTTS(config)
        
        # Initialize TTS (but don't actually play audio)
        success = tts._init_piper()
        
        if success:
            return True, "TTS initialized successfully"
        else:
            return False, "Failed to initialize TTS"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def test_llm() -> Tuple[bool, str]:
    """
    Test LLM initialization.
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    try:
        from maggie import LLMProcessor
        
        # Basic LLM configuration
        config = {
            "model_path": "models/mistral-7b-instruct-v0.3-GPTQ-4bit",
            "model_type": "mistral",
            "gpu_layers": 32
        }
        
        # Check if model directory exists
        if not os.path.exists(config["model_path"]):
            return False, f"LLM model directory not found: {config['model_path']}"
        
        # Create LLM processor
        llm = LLMProcessor(config)
        
        # Don't actually load the model, just return success
        return True, "LLM model directory verified"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def test_wake_word() -> Tuple[bool, str]:
    """
    Test wake word detector initialization.
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    try:
        from maggie import WakeWordDetector
        
        # Basic wake word configuration
        config = {
            "sensitivity": 0.5,
            "keyword_path": None,
            "porcupine_access_key": ""
        }
        
        # Check if Porcupine access key is set in config.yaml
        try:
            import yaml
            with open("config.yaml", "r") as f:
                full_config = yaml.safe_load(f)
                access_key = full_config.get("wake_word", {}).get("porcupine_access_key", "")
                if access_key:
                    config["porcupine_access_key"] = access_key
        except Exception:
            pass
        
        if not config["porcupine_access_key"]:
            return False, "Porcupine access key not set in config.yaml"
        
        # Create wake word detector but don't start it
        detector = WakeWordDetector(config)
        
        return True, "Wake word detector can be initialized"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Maggie AI Assistant Test Script")
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--config", 
        action="store_true",
        help="Test configuration loading"
    )
    parser.add_argument(
        "--event-bus", 
        action="store_true",
        help="Test event bus functionality"
    )
    parser.add_argument(
        "--gui", 
        action="store_true",
        help="Test GUI initialization"
    )
    parser.add_argument(
        "--tts", 
        action="store_true",
        help="Test text-to-speech functionality"
    )
    parser.add_argument(
        "--llm", 
        action="store_true",
        help="Test LLM initialization"
    )
    parser.add_argument(
        "--wake-word", 
        action="store_true",
        help="Test wake word detector initialization"
    )
    return parser.parse_args()

def main() -> int:
    """
    Run selected tests.
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    args = parse_arguments()
    
    # If no tests specified, run config test only
    if not any([args.all, args.config, args.event_bus, args.gui, args.tts, args.llm, args.wake_word]):
        print("No tests specified, running configuration test only")
        args.config = True
    
    tests = []
    
    if args.all or args.config:
        tests.append(("Configuration Loading", test_config_loading))
    
    if args.all or args.event_bus:
        tests.append(("Event Bus", test_event_bus))
    
    if args.all or args.gui:
        tests.append(("GUI Initialization", test_gui_initialization))
    
    if args.all or args.tts:
        tests.append(("Text-to-Speech", test_tts))
    
    if args.all or args.llm:
        tests.append(("LLM Initialization", test_llm))
    
    if args.all or args.wake_word:
        tests.append(("Wake Word Detector", test_wake_word))
    
    results = []
    for name, func in tests:
        success, message = run_test(name, func)
        results.append((name, success, message))
    
    print("\n=== Test Results ===")
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, message in results:
        status = "PASSED" if success else "FAILED"
        print(f"{status}: {name} - {message}")
    
    print(f"\nPassed {passed} of {total} tests")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
=======
"""
Maggie AI Assistant - Test Script
==============================
Tests critical components of the Maggie AI Assistant.
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional

def run_test(test_name: str, test_func, *args, **kwargs) -> Tuple[bool, str]:
    """
    Run a test function and return the result.
    
    Parameters
    ----------
    test_name : str
        Name of the test
    test_func : callable
        Test function to run
    *args, **kwargs
        Arguments to pass to the test function
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    print(f"Running test: {test_name}...")
    try:
        start_time = time.time()
        result = test_func(*args, **kwargs)
        elapsed = time.time() - start_time
        if isinstance(result, tuple) and len(result) >= 2:
            success, message = result[0], result[1]
        else:
            success, message = result, "Test completed successfully"
        
        status = "PASSED" if success else "FAILED"
        print(f"{status}: {test_name} ({elapsed:.2f}s) - {message}")
        return success, message
    except Exception as e:
        print(f"FAILED: {test_name} - Exception: {str(e)}")
        return False, str(e)

def test_config_loading() -> Tuple[bool, str]:
    """
    Test loading configuration from file.
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    try:
        from utils.config import Config
        
        config = Config().load()
        if not config:
            return False, "Failed to load configuration"
        
        # Check for required sections
        required_sections = ["wake_word", "speech", "llm"]
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            return False, f"Missing required sections: {', '.join(missing_sections)}"
        
        return True, "Configuration loaded successfully"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def test_event_bus() -> Tuple[bool, str]:
    """
    Test the event bus functionality.
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    try:
        from maggie import EventBus
        
        event_bus = EventBus()
        received_events = []
        
        def test_handler(data):
            received_events.append(data)
        
        # Subscribe to test event
        event_bus.subscribe("test_event", test_handler)
        
        # Start event bus
        event_bus.start()
        
        # Publish test event
        event_bus.publish("test_event", "test_data")
        
        # Wait for event processing
        time.sleep(0.5)
        
        # Stop event bus
        event_bus.stop()
        
        if len(received_events) == 1 and received_events[0] == "test_data":
            return True, "Event bus working correctly"
        else:
            return False, f"Event not received correctly, got: {received_events}"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def test_gui_initialization() -> Tuple[bool, str]:
    """
    Test GUI initialization.
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    try:
        from PyQt6.QtWidgets import QApplication
        from utils.gui import MainWindow
        
        # Create QApplication instance
        app = QApplication([])
        
        # Create a mock Maggie object
        class MockMaggie:
            def __init__(self):
                self.utilities = {}
                
            def on_shutdown_clicked(self):
                pass
                
            def on_sleep_clicked(self):
                pass
                
            def process_command(self, utility=None):
                pass
        
        # Create MainWindow
        window = MainWindow(MockMaggie())
        
        # Check if window was created
        if window is not None:
            return True, "GUI initialized successfully"
        else:
            return False, "Failed to initialize GUI"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def test_tts() -> Tuple[bool, str]:
    """
    Test text-to-speech functionality.
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    try:
        from utils.tts import PiperTTS
        
        # Basic TTS configuration
        config = {
            "voice_model": "en_US-kathleen-medium",
            "model_path": "models/tts"
        }
        
        # Check if model files exist
        model_dir = os.path.join(config["model_path"], config["voice_model"])
        onnx_path = os.path.join(model_dir, f"{config['voice_model']}.onnx")
        json_path = os.path.join(model_dir, f"{config['voice_model']}.json")
        
        if not os.path.exists(onnx_path) or not os.path.exists(json_path):
            return False, f"TTS model files not found at {model_dir}"
        
        # Create TTS instance
        tts = PiperTTS(config)
        
        # Initialize TTS (but don't actually play audio)
        success = tts._init_piper()
        
        if success:
            return True, "TTS initialized successfully"
        else:
            return False, "Failed to initialize TTS"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def test_llm() -> Tuple[bool, str]:
    """
    Test LLM initialization.
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    try:
        from maggie import LLMProcessor
        
        # Basic LLM configuration
        config = {
            "model_path": "models/mistral-7b-instruct-v0.3-GPTQ-4bit",
            "model_type": "mistral",
            "gpu_layers": 32
        }
        
        # Check if model directory exists
        if not os.path.exists(config["model_path"]):
            return False, f"LLM model directory not found: {config['model_path']}"
        
        # Create LLM processor
        llm = LLMProcessor(config)
        
        # Don't actually load the model, just return success
        return True, "LLM model directory verified"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def test_wake_word() -> Tuple[bool, str]:
    """
    Test wake word detector initialization.
    
    Returns
    -------
    Tuple[bool, str]
        Success status and result message
    """
    try:
        from maggie import WakeWordDetector
        
        # Basic wake word configuration
        config = {
            "sensitivity": 0.5,
            "keyword_path": None,
            "porcupine_access_key": ""
        }
        
        # Check if Porcupine access key is set in config.yaml
        try:
            import yaml
            with open("config.yaml", "r") as f:
                full_config = yaml.safe_load(f)
                access_key = full_config.get("wake_word", {}).get("porcupine_access_key", "")
                if access_key:
                    config["porcupine_access_key"] = access_key
        except Exception:
            pass
        
        if not config["porcupine_access_key"]:
            return False, "Porcupine access key not set in config.yaml"
        
        # Create wake word detector but don't start it
        detector = WakeWordDetector(config)
        
        return True, "Wake word detector can be initialized"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Maggie AI Assistant Test Script")
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--config", 
        action="store_true",
        help="Test configuration loading"
    )
    parser.add_argument(
        "--event-bus", 
        action="store_true",
        help="Test event bus functionality"
    )
    parser.add_argument(
        "--gui", 
        action="store_true",
        help="Test GUI initialization"
    )
    parser.add_argument(
        "--tts", 
        action="store_true",
        help="Test text-to-speech functionality"
    )
    parser.add_argument(
        "--llm", 
        action="store_true",
        help="Test LLM initialization"
    )
    parser.add_argument(
        "--wake-word", 
        action="store_true",
        help="Test wake word detector initialization"
    )
    return parser.parse_args()

def main() -> int:
    """
    Run selected tests.
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    args = parse_arguments()
    
    # If no tests specified, run config test only
    if not any([args.all, args.config, args.event_bus, args.gui, args.tts, args.llm, args.wake_word]):
        print("No tests specified, running configuration test only")
        args.config = True
    
    tests = []
    
    if args.all or args.config:
        tests.append(("Configuration Loading", test_config_loading))
    
    if args.all or args.event_bus:
        tests.append(("Event Bus", test_event_bus))
    
    if args.all or args.gui:
        tests.append(("GUI Initialization", test_gui_initialization))
    
    if args.all or args.tts:
        tests.append(("Text-to-Speech", test_tts))
    
    if args.all or args.llm:
        tests.append(("LLM Initialization", test_llm))
    
    if args.all or args.wake_word:
        tests.append(("Wake Word Detector", test_wake_word))
    
    results = []
    for name, func in tests:
        success, message = run_test(name, func)
        results.append((name, success, message))
    
    print("\n=== Test Results ===")
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, message in results:
        status = "PASSED" if success else "FAILED"
        print(f"{status}: {name} - {message}")
    
    print(f"\nPassed {passed} of {total} tests")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
>>>>>>> 6062514b96de23fbf6dcdbfd4420d6e2f22903ff
    sys.exit(main())