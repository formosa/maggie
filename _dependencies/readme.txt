# Maggie AI Assistant

## Assets
These application dependies are provided as a convenience.
- **Python 3.10.11 Windows 64bit Installer**: python-3.10.11-amd64.exe
- **CUDA Toolkit 11.8 for Windows 11 64bit Installer**: cuda_11.8.0_522.06_windows.exe
- **cuDNN Backend for Windows**:  cudnn-windows-x86_64-9.8.0.87_cuda11-archive.zip
  - Download the zip from: [https://developer.nvidia.com/cuda-11-8-0-download-archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)
    - Unzip the cuDNN package
    - Copy the following files from the unzipped package into the NVIDIA cuDNN directory
    - Copy bin\cudnn*.dll to C:\Program Files\NVIDIA\CUDNN\v9.x\bin
    - Copy include\cudnn*.h to C:\Program Files\NVIDIA\CUDNN\v9.x\include
    - Copy lib\x64\cudnn*.lib to C:\Program Files\NVIDIA\CUDNN\v9.x\lib
    - Set the following environment variable to point to where cuDNN is located. 
      To access the value of the $(PATH) environment variable, perform the following steps:
        - Open a command prompt from the Start menu.
        - Type Run and hit Enter.
        - Issue the control sysdm.cpl command.
        - Select the Advanced tab at the top of the window.
        - Click Environment Variables at the bottom of the window.
        - Add the NVIDIA cuDNN bin directory path to the PATH variable:
            - Variable Name: PATH
            - Value to Add: C:\Program Files\NVIDIA\CUDNN\v9.x\bin
    - Or, install using pip:
        - Update pip and wheel: py -m pip install --upgrade pip wheel
        - To install cuDNN for CUDA 11, run: py -m pip install nvidia-cudnn-cu11
- **Visual C++ Redistributable**: VC_redist.x64.exe
    - Windows:
        - [https://aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- **Porcupine Wake Word Engine License**
   - [https://console.picovoice.ai/](https://console.picovoice.ai/)