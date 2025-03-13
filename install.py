import os
import sys
import subprocess
import site
import shutil
import zipfile
import urllib.request

# Base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_command(command, cwd=None):
    """
    Run a shell command and handle errors.

    This function executes the given shell command in the specified working directory.
    If the command fails, it prints an error message and exits the program.

    Parameters
    ----------
    command : str
        The shell command to execute.
    cwd : str, optional
        The working directory in which to run the command. Defaults to None.

    Raises
    ------
    subprocess.CalledProcessError
        If the command returns a non-zero exit status.

    Examples
    --------
    >>> run_command("echo Hello, World!")
    Hello, World!
    """
    try:
        subprocess.run(command, check=True, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(e)
        sys.exit(1)

def download_and_extract(url, dest_dir):
    """
    Download a ZIP file and extract it to a destination directory.

    This function downloads a ZIP file from the given URL and extracts its contents
    to the specified destination directory. It also cleans up the ZIP file after extraction.

    Parameters
    ----------
    url : str
        The URL of the ZIP file to download.
    dest_dir : str
        The directory where the ZIP file will be extracted.

    Returns
    -------
    str
        The path to the extracted directory (assumed to be the first top-level folder).

    Examples
    --------
    >>> extracted_dir = download_and_extract("https://example.com/file.zip", "downloads")
    Downloading https://example.com/file.zip...
    Extracting downloads/file.zip...
    """
    zip_path = os.path.join(BASE_DIR, "downloads", os.path.basename(url))
    extract_dir = os.path.join(BASE_DIR, "downloads")
    
    # Create downloads directory if it doesn't exist
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    os.remove(zip_path)  # Clean up ZIP file
    
    # Return the extracted directory name (assuming it's the first top-level folder)
    extracted_folder = os.path.join(extract_dir, os.listdir(extract_dir)[0])
    return extracted_folder

def patch_setup_py(setup_py_path, espeak_dir, onnx_include, onnx_lib):
    """
    Patch piper-phonemize's setup.py with correct include and library paths.

    This function modifies the `setup.py` file of `piper-phonemize` to use the correct
    paths for `espeak-ng` and `onnxruntime` on Windows.

    Parameters
    ----------
    setup_py_path : str
        The path to the `setup.py` file to patch.
    espeak_dir : str
        The directory where `espeak-ng` is built.
    onnx_include : str
        The include directory for `onnxruntime`.
    onnx_lib : str
        The library directory for `onnxruntime`.

    Examples
    --------
    >>> patch_setup_py("path/to/setup.py", "path/to/espeak-ng", "path/to/onnx/include", "path/to/onnx/lib")
    Patching setup.py for Windows compatibility...
    """
    with open(setup_py_path, "r") as f:
        content = f.read()
    
    # Define new paths (escaping backslashes for Windows)
    espeak_include = os.path.join(espeak_dir, "src", "include").replace("\\", "\\\\")
    espeak_lib = os.path.join(espeak_dir, "Release").replace("\\", "\\\\")
    onnx_include = onnx_include.replace("\\", "\\\\")
    onnx_lib = onnx_lib.replace("\\", "\\\\")
    
    # Replace the include_dirs, library_dirs, and libraries lines
    new_include_dirs = f'        include_dirs=["{espeak_include}", "{onnx_include}"],'
    new_library_dirs = f'        library_dirs=["{espeak_lib}", "{onnx_lib}"],'
    new_libraries = '        libraries=["espeak-ng", "onnxruntime"],'
    
    # Find and replace the relevant lines (assuming they're in Pybind11Extension)
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if "include_dirs=[" in line:
            lines[i] = new_include_dirs
        elif "library_dirs=[" in line:
            lines[i] = new_library_dirs
        elif "libraries=[" in line:
            lines[i] = new_libraries
    
    with open(setup_py_path, "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    """
    Main installation script for the Maggie project.

    This script handles the installation of all dependencies required for the Maggie project,
    including PyTorch, standard dependencies, and piper-phonemize. It is designed to work on
    both Windows and non-Windows systems, with special handling for Windows to ensure
    compatibility with piper-phonemize.

    The script performs the following steps:
    1. Activates the virtual environment.
    2. Installs PyTorch with CUDA support.
    3. Installs standard dependencies from temp_requirements.txt.
    4. Installs onnxruntime (Windows only).
    5. Downloads, builds, and installs piper-phonemize with necessary patches (Windows only).
    6. Cleans up temporary files.

    On non-Windows systems, it installs piper-phonemize directly from GitHub.

    Prerequisites:
    - Python 3.10 with a virtual environment created at 'venv'.
    - Visual Studio Build Tools 2022 with C++ workload (Windows only).
    - Internet access for downloading dependencies.

    Usage:
    - Run this script from the project root after activating the virtual environment.
    """
    # Activate virtual environment if present
    venv_pip = os.path.join(BASE_DIR, "venv", "Scripts", "pip.exe" if sys.platform == "win32" else "bin", "pip")
    python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe" if sys.platform == "win32" else "bin", "python")
    
    if not os.path.exists(venv_pip):
        print("Virtual environment not found. Please create one first.")
        sys.exit(1)

    if sys.platform == "win32":
        # Path to vcvarsall.bat (adjust if your Visual Studio installation differs)
        vcvarsall = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
        if not os.path.exists(vcvarsall):
            print("Visual Studio Build Tools not found. Please install them with C++ workload.")
            sys.exit(1)

        # Install PyTorch with CUDA
        print("Installing PyTorch...")
        run_command(
            f'cmd /c "{vcvarsall} x64 && {venv_pip} install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118"'
        )

        # Install standard dependencies
        requirements_file = os.path.join(BASE_DIR, "temp_requirements.txt")
        if os.path.exists(requirements_file):
            print("Installing standard dependencies...")
            run_command(
                f'cmd /c "{vcvarsall} x64 && {venv_pip} install -r {requirements_file}"'
            )
        else:
            print("Warning: temp_requirements.txt not found. Skipping standard dependencies.")

        # Install onnxruntime
        print("Installing onnxruntime...")
        run_command(
            f'cmd /c "{vcvarsall} x64 && {venv_pip} install onnxruntime}"'
        )

        # Download and extract piper-phonemize
        piper_url = "https://github.com/rhasspy/piper-phonemize/archive/refs/heads/master.zip"
        piper_dir = download_and_extract(piper_url, os.path.join(BASE_DIR, "downloads"))
        piper_dir = os.path.join(BASE_DIR, "downloads", "piper-phonemize-master")  # Adjust based on extracted name

        # Build espeak-ng
        espeak_dir = os.path.join(piper_dir, "espeak-ng")
        print("Building espeak-ng...")
        espeak_project = os.path.join(espeak_dir, "espeak-ng.vcxproj")
        if not os.path.exists(espeak_project):
            print("Error: espeak-ng.vcxproj not found in piper-phonemize repository.")
            sys.exit(1)
        run_command(
            f'cmd /c "{vcvarsall} x64 && msbuild {espeak_project} /p:Configuration=Release /p:Platform=x64"',
            cwd=espeak_dir
        )

        # Patch setup.py
        setup_py_path = os.path.join(piper_dir, "setup.py")
        site_packages = site.getsitepackages()[0]
        onnx_include = os.path.join(site_packages, "onnxruntime", "capi", "include")
        onnx_lib = os.path.join(site_packages, "onnxruntime", "capi")
        print("Patching setup.py for Windows compatibility...")
        patch_setup_py(setup_py_path, espeak_dir, onnx_include, onnx_lib)

        # Install piper-phonemize from modified source
        print("Installing piper-phonemize from source...")
        run_command(
            f'cmd /c "{vcvarsall} x64 && {venv_pip} install {piper_dir}"'
        )

        # Clean up downloads
        shutil.rmtree(os.path.join(BASE_DIR, "downloads"), ignore_errors=True)

    else:
        # Non-Windows installation (simplified)
        print("Installing PyTorch...")
        run_command(f"{venv_pip} install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118")

        print("Installing standard dependencies...")
        requirements_file = os.path.join(BASE_DIR, "temp_requirements.txt")
        if os.path.exists(requirements_file):
            run_command(f"{venv_pip} install -r {requirements_file}")
        else:
            print("Warning: temp_requirements.txt not found. Skipping standard dependencies.")

        print("Installing piper-phonemize from GitHub...")
        run_command(f"{venv_pip} install git+https://github.com/rhasspy/piper-phonemize.git")

    print("Installation completed successfully!")