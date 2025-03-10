# Maggie AI Assistant - Command Reference

This document provides a comprehensive reference of all voice commands and their functions in the Maggie AI Assistant.

## Wake Word

| Command | State | Description |
|---------|-------|-------------|
| "Maggie" | IDLE | Activates the assistant, transitions from IDLE to STARTUP state |

## Core System Commands

| Command | Valid States | Description |
|---------|-------------|-------------|
| "Sleep" or "Go to sleep" | READY, ACTIVE | Puts Maggie into IDLE state to conserve resources |
| "Shutdown" or "Turn off" | Any | Initiates the shutdown process to close the application |
| "Cancel" | ACTIVE | Cancels the current operation |

## Utility Commands

| Command | Valid States | Description |
|---------|-------------|-------------|
| "New recipe" | READY | Activates the Recipe Creator utility |

## Recipe Creator Commands

| Command | Context | Description |
|---------|---------|-------------|
| "[Recipe name]" | When prompted for recipe name | Provides a name for the new recipe |
| "Yes" or "Correct" | During name confirmation | Confirms the recognized recipe name is correct |
| "No" or "Wrong" | During name confirmation | Indicates the recognized recipe name is incorrect |
| "[Recipe description]" | When prompted for recipe description | Provides ingredients and preparation steps for the recipe |
| "Cancel" | Any time during recipe creation | Cancels the recipe creation process |

## GUI Controls

| Control | Description |
|---------|-------------|
| Sleep Button | Equivalent to saying "Sleep" |
| Shutdown Button | Equivalent to saying "Shutdown" |
| Recipe Creator Button | Equivalent to saying "New recipe" |
| Chat Log Tab | View conversation history |
| Event Log Tab | View system events and state transitions |
| Error Log Tab | View error messages |

## Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--config` | Specify path to configuration file | `python main.py --config custom_config.yaml` |
| `--debug` | Enable debug logging | `python main.py --debug` |

## State Transitions

| Current State | Command/Event | Next State |
|---------------|---------------|------------|
| IDLE | "Maggie" (wake word) | STARTUP |
| STARTUP | Initialization complete | READY |
| READY | "New recipe" | ACTIVE |
| ACTIVE | Recipe completion | READY |
| READY | Inactivity timeout | CLEANUP → IDLE |
| Any | "Shutdown" | CLEANUP → SHUTDOWN |
| Any | "Sleep" | CLEANUP → IDLE |

## Response Expectations

| Command | Expected Response |
|---------|------------------|
| "Maggie" (wake word) | "Initializing Maggie" followed by "Ready for your command" |
| "New recipe" | "Starting recipe creator. Let's create a new recipe." |
| "Sleep" | "Going to sleep" |
| "Shutdown" | "Shutting down" |

## Configuration Parameters

Key configuration parameters that affect command behavior:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `wake_word.sensitivity` | Sensitivity of wake word detection (0.0-1.0) | 0.5 |
| `inactivity_timeout` | Seconds of inactivity before READY → IDLE transition | 300 |
| `speech.whisper.model_size` | Size of Whisper model for speech recognition | "base" |

## Error Handling Commands

| Situation | Recommended Command |
|-----------|---------------------|
| Incorrect recipe name recognized | "No" or "Wrong" |
| Need to stop current operation | "Cancel" |
| System unresponsive | Click Shutdown button, restart application |

## Tips for Effective Commands

- **Speak clearly** at a normal pace and volume
- **Wait for state transitions** before giving the next command
- **Check current state** in the GUI before giving commands
- **Use specific command phrases** exactly as listed
- **Allow a brief pause** after saying the wake word
