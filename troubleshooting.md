<<<<<<< HEAD
# Maggie AI Assistant - Troubleshooting Guide

This guide provides solutions for common issues encountered with the Maggie AI Assistant.

## Initial Setup Issues

### Python Version Problems

**Symptom:** Error message about incompatible Python version.

**Solution:**
1. Verify your Python version by running `python --version`
2. If not using Python 3.10.x, install it from [python.org](https://www.python.org/downloads/)
3. On Windows, ensure you check "Add Python to PATH" during installation
4. Create a new virtual environment with Python 3.10

### Dependency Installation Failures

**Symptom:** Errors when installing requirements with pip.

**Solution:**
1. For PyAudio errors on Windows:
   ```powershell
   pip install pipwin
=======
# Maggie AI Assistant - Troubleshooting Guide

This guide provides solutions for common issues encountered with the Maggie AI Assistant.

## Initial Setup Issues

### Python Version Problems

**Symptom:** Error message about incompatible Python version.

**Solution:**
1. Verify your Python version by running `python --version`
2. If not using Python 3.10.x, install it from [python.org](https://www.python.org/downloads/)
3. On Windows, ensure you check "Add Python to PATH" during installation
4. Create a new virtual environment with Python 3.10

### Dependency Installation Failures

**Symptom:** Errors when installing requirements with pip.

**Solution:**
1. For PyAudio errors on Windows:
   ```powershell
   pip install pipwin
>>>>>>> 6062514b96de23fbf6dcdbfd4420d6e2f22903ff
   pipwin install pyaudio