import argparse,json,os,platform,shutil,subprocess,sys,time,urllib.request,zipfile
from pathlib import Path
from typing import Dict,Any,List,Tuple,Optional,Union,Callable
class ColorOutput:
	def __init__(self,force_enable:bool=False):
		self.enabled=force_enable or self._supports_color()
		if self.enabled:self.colors={'reset':'\x1b[0m','bold':'\x1b[1m','red':'\x1b[91m','green':'\x1b[92m','yellow':'\x1b[93m','blue':'\x1b[94m','magenta':'\x1b[95m','cyan':'\x1b[96m','white':'\x1b[97m'}
		else:self.colors={color:''for color in['reset','bold','red','green','yellow','blue','magenta','cyan','white']}
	def _supports_color(self)->bool:
		if platform.system()=='Windows':
			if int(platform.release())>=10:return True
			return False
		if os.environ.get('NO_COLOR'):return False
		if os.environ.get('FORCE_COLOR'):return True
		return hasattr(sys.stdout,'isatty')and sys.stdout.isatty()
	def print(self,message:str,color:Optional[str]=None,bold:bool=False):
		formatted=message
		if self.enabled:
			if bold and'bold'in self.colors:formatted=f"{self.colors['bold']}{formatted}"
			if color and color in self.colors:formatted=f"{self.colors[color]}{formatted}"
			if(bold or color)and'reset'in self.colors:formatted=f"{formatted}{self.colors['reset']}"
		print(formatted)
	def input(self,prompt:str,color:Optional[str]=None,bold:bool=False)->str:
		formatted=prompt
		if self.enabled:
			if bold and'bold'in self.colors:formatted=f"{self.colors['bold']}{formatted}"
			if color and color in self.colors:formatted=f"{self.colors[color]}{formatted}"
			if(bold or color)and'reset'in self.colors:formatted=f"{formatted}{self.colors['reset']}"
		return input(formatted)
class ProgressTracker:
	def __init__(self,color:ColorOutput,total_steps:int=10):self.color=color;self.total_steps=total_steps;self.current_step=0;self.start_time=time.time()
	def start_step(self,step_name:str):self.current_step+=1;elapsed=time.time()-self.start_time;self.color.print(f"\n[{self.current_step}/{self.total_steps}] {step_name} (Elapsed: {elapsed:.1f}s)",color='cyan',bold=True)
	def complete_step(self,success:bool=True,message:Optional[str]=None):
		if success:status='✓ Complete';color='green'
		else:status='✗ Failed';color='red'
		msg=f"  {status}"
		if message:msg+=f": {message}"
		self.color.print(msg,color=color)
	def elapsed_time(self)->float:return time.time()-self.start_time
	def display_summary(self,success:bool=True):
		elapsed=self.elapsed_time()
		if success:status='Installation Completed Successfully';color='green'
		else:status='Installation Completed with Errors';color='yellow'
		self.color.print(f"\n=== {status} ===",color=color,bold=True);self.color.print(f"Total time: {elapsed:.1f} seconds")
class MaggieInstaller:
	def __init__(self,verbose:bool=False,cpu_only:bool=False,skip_models:bool=False,skip_problematic:bool=False,force_reinstall:bool=False):self.verbose=verbose;self.cpu_only=cpu_only;self.skip_models=skip_models;self.skip_problematic=skip_problematic;self.force_reinstall=force_reinstall;self.base_dir=Path(os.path.dirname(os.path.abspath(__file__)));self.platform_system=platform.system();self.platform_machine=platform.machine();self.required_dirs=['downloads','logs','maggie','maggie/cache','maggie/cache/tts','maggie/core','maggie/extensions','maggie/models','maggie/models/llm','maggie/models/stt','maggie/models/tts','maggie/templates','maggie/templates/extension','maggie/utils','maggie/utils/hardware','maggie/utils/config','maggie/utils/llm','maggie/utils/stt','maggie/utils/tts'];self.color=ColorOutput();self.total_steps=8;self.progress=ProgressTracker(self.color,self.total_steps);self.is_admin=self._check_admin_privileges();self.has_git=False;self.has_cpp_compiler=False;self.hardware_info={'cpu':{'is_ryzen_9_5900x':False,'model':'','cores':0,'threads':0},'gpu':{'is_rtx_3080':False,'model':'','vram_gb':0,'cuda_available':False,'cuda_version':''},'memory':{'total_gb':0,'available_gb':0,'is_32gb':False}}
	def _check_admin_privileges(self)->bool:
		try:
			if self.platform_system=='Windows':import ctypes;return ctypes.windll.shell32.IsUserAnAdmin()!=0
			else:return os.geteuid()==0
		except:return False
	def _run_command(self,command:List[str],check:bool=True,shell:bool=False,capture_output:bool=True,cwd:Optional[str]=None)->Tuple[int,str,str]:
		if self.verbose:self.color.print(f"Running command: {' '.join(command)}",'cyan')
		try:
			if capture_output:process=subprocess.Popen(command if not shell else' '.join(command),stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=shell,text=True,cwd=cwd);stdout,stderr=process.communicate();return_code=process.returncode
			else:process=subprocess.Popen(command if not shell else' '.join(command),shell=shell,cwd=cwd);process.communicate();return_code=process.returncode;stdout,stderr='',''
			if check and return_code!=0 and capture_output:
				if self.verbose:self.color.print(f"Command failed with code {return_code}: {stderr}",'red')
			return return_code,stdout,stderr
		except Exception as e:
			if self.verbose:self.color.print(f"Error executing command: {e}",'red')
			return-1,'',str(e)
	def _download_file(self,url:str,destination:str,show_progress:bool=True)->bool:
		try:
			self.color.print(f"Downloading {url}",'blue');os.makedirs(os.path.dirname(destination),exist_ok=True)
			with urllib.request.urlopen(url)as response,open(destination,'wb')as out_file:
				file_size=int(response.info().get('Content-Length',0));downloaded=0;block_size=8192*8
				if show_progress and file_size>0:self.color.print(f"Total file size: {file_size/1024/1024:.1f} MB")
				last_percent=0
				while True:
					buffer=response.read(block_size)
					if not buffer:break
					downloaded+=len(buffer);out_file.write(buffer)
					if show_progress and file_size>0:
						percent=int(downloaded*100/file_size)
						# Only print progress every 10%
						if percent>=last_percent+10:
							last_percent=percent
							self.color.print(f"  Progress: {percent}% ({downloaded/1024/1024:.1f}/{file_size/1024/1024:.1f} MB)")
			self.color.print(f"Download completed: {destination}",'green');return True
		except Exception as e:self.color.print(f"Error downloading file: {e}",'red');return False
	def _verify_python_version(self)->bool:
		version=platform.python_version_tuple()
		if int(version[0])!=3 or int(version[1])!=10:
			self.color.print(f"ERROR: Incompatible Python version: {platform.python_version()}",'red',bold=True);self.color.print('Maggie requires Python 3.10.x specifically. Other versions are not supported.','red')
			if self.platform_system=='Windows':self.color.print('Please install Python 3.10 from: https://www.python.org/downloads/release/python-31011/','yellow')
			else:self.color.print('Please install Python 3.10 using:','yellow');self.color.print('sudo apt install python3.10 python3.10-venv python3.10-dev','yellow')
			return False
		self.color.print(f"Python {platform.python_version()} - Compatible ✓",'green');return True
	def _detect_hardware(self)->Dict[str,Any]:
		hardware_info={'cpu':self._detect_cpu(),'memory':self._detect_memory(),'gpu':self._detect_gpu()if not self.cpu_only else{'available':False}};self.color.print('Hardware Detection:','cyan',bold=True);cpu_info=hardware_info['cpu']
		if cpu_info['is_ryzen_9_5900x']:self.color.print('  CPU: AMD Ryzen 9 5900X detected ✓','green')
		else:self.color.print(f"  CPU: {cpu_info['model']}",'yellow');self.color.print(f"       {cpu_info['cores']} cores / {cpu_info['threads']} threads",'yellow')
		memory_info=hardware_info['memory']
		if memory_info['is_32gb']:self.color.print(f"  RAM: {memory_info['total_gb']:.1f} GB (32GB detected) ✓",'green')
		else:self.color.print(f"  RAM: {memory_info['total_gb']:.1f} GB",'yellow')
		gpu_info=hardware_info['gpu']
		if self.cpu_only:self.color.print('  GPU: CPU-only mode selected (skipping GPU detection)','yellow')
		elif gpu_info['is_rtx_3080']:
			self.color.print(f"  GPU: NVIDIA RTX 3080 detected ✓",'green');self.color.print(f"       {gpu_info['vram_gb']:.1f} GB VRAM",'green')
			if gpu_info['cuda_available']:self.color.print(f"       CUDA {gpu_info['cuda_version']} available",'green')
		elif gpu_info['available']:
			self.color.print(f"  GPU: {gpu_info['model']}",'yellow');self.color.print(f"       {gpu_info['vram_gb']:.1f} GB VRAM",'yellow')
			if gpu_info['cuda_available']:self.color.print(f"       CUDA {gpu_info['cuda_version']} available",'yellow')
		else:self.color.print('  GPU: No compatible GPU detected','red')
		return hardware_info
	def _detect_cpu(self)->Dict[str,Any]:
		cpu_info={'is_ryzen_9_5900x':False,'model':platform.processor()or'Unknown','cores':0,'threads':0}
		try:import psutil;cpu_info['cores']=psutil.cpu_count(logical=False)or 0;cpu_info['threads']=psutil.cpu_count(logical=True)or 0
		except ImportError:cpu_info['threads']=os.cpu_count()or 0;cpu_info['cores']=cpu_info['threads']//2
		model_lower=cpu_info['model'].lower()
		if'ryzen 9'in model_lower and'5900x'in model_lower:cpu_info['is_ryzen_9_5900x']=True
		if self.platform_system=='Windows':
			try:
				import wmi;c=wmi.WMI()
				for cpu in c.Win32_Processor():
					cpu_info['model']=cpu.Name
					if'Ryzen 9 5900X'in cpu.Name:cpu_info['is_ryzen_9_5900x']=True
					break
			except:pass
		return cpu_info
	def _detect_memory(self)->Dict[str,Any]:
		memory_info={'total_gb':0,'available_gb':0,'is_32gb':False,'type':'Unknown'}
		try:
			import psutil;mem=psutil.virtual_memory();memory_info['total_gb']=mem.total/1024**3;memory_info['available_gb']=mem.available/1024**3;memory_info['is_32gb']=30<=memory_info['total_gb']<=34
			if self.platform_system=='Windows':
				try:
					import wmi;c=wmi.WMI()
					for mem_module in c.Win32_PhysicalMemory():
						if hasattr(mem_module,'PartNumber')and mem_module.PartNumber:
							if'DDR4'in mem_module.PartNumber:
								memory_info['type']='DDR4'
								if'3200'in mem_module.PartNumber:memory_info['type']='DDR4-3200'
								break
				except:pass
		except ImportError:memory_info['total_gb']=0;memory_info['available_gb']=0
		return memory_info
	def _detect_gpu(self)->Dict[str,Any]:
		gpu_info={'available':False,'is_rtx_3080':False,'model':'Unknown','vram_gb':0,'cuda_available':False,'cuda_version':''}
		if self.cpu_only:return gpu_info
		try:
			returncode,stdout,_=self._run_command([sys.executable,'-c',"import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None; print(f'CUDA Version: {torch.version.cuda}') if torch.cuda.is_available() else None"],check=False)
			if returncode!=0:
				if self.verbose:self.color.print('PyTorch not installed yet, CUDA status unknown','yellow')
				return gpu_info
			for line in stdout.splitlines():
				if'CUDA: True'in line:gpu_info['available']=True;gpu_info['cuda_available']=True
				elif'Device:'in line:
					gpu_info['model']=line.split('Device:')[1].strip()
					if'3080'in gpu_info['model']:gpu_info['is_rtx_3080']=True
				elif'CUDA Version:'in line:gpu_info['cuda_version']=line.split('CUDA Version:')[1].strip()
			if gpu_info['cuda_available']:
				returncode,stdout,_=self._run_command([sys.executable,'-c',"import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}')"],check=False)
				if returncode==0 and'VRAM:'in stdout:
					vram_str=stdout.split('VRAM:')[1].strip()
					try:gpu_info['vram_gb']=float(vram_str)
					except:pass
		except:pass
		if self.verbose and not gpu_info['available']and not self.cpu_only:self.color.print('No CUDA-capable GPU detected or PyTorch not installed','yellow');self.color.print('Consider using --cpu-only if no GPU is available','yellow')
		return gpu_info
	def _check_tools(self)->Dict[str,bool]:
		tools_status={'git':False,'cpp_compiler':False};returncode,stdout,_=self._run_command(['git','--version'],check=False)
		if returncode==0:tools_status['git']=True;self.has_git=True;self.color.print(f"Git found: {stdout.strip()}",'green')
		else:
			self.color.print('Git not found - limited functionality will be available','yellow');self.color.print('Install Git for full functionality:','yellow')
			if self.platform_system=='Windows':self.color.print('https://git-scm.com/download/win','yellow')
			else:self.color.print('sudo apt-get install git','yellow')
		if self.platform_system=='Windows':
			returncode,_,_=self._run_command(['where','cl.exe'],check=False)
			if returncode==0:tools_status['cpp_compiler']=True;self.has_cpp_compiler=True;self.color.print('Visual C++ compiler (cl.exe) found','green')
			else:
				self.color.print('Visual C++ compiler not found','yellow');self.color.print('Some packages may need to be installed from wheels','yellow')
				if self.verbose:self.color.print('Install Visual C++ Build Tools:','yellow');self.color.print('https://visualstudio.microsoft.com/visual-cpp-build-tools/','yellow')
		else:
			returncode,_,_=self._run_command(['which','gcc'],check=False)
			if returncode==0:tools_status['cpp_compiler']=True;self.has_cpp_compiler=True;self.color.print('GCC compiler found','green')
			else:
				self.color.print('GCC compiler not found','yellow');self.color.print('Some packages may fail to build','yellow')
				if self.verbose:self.color.print('Install build tools:','yellow');self.color.print('sudo apt-get install build-essential','yellow')
		return tools_status
	def _create_directories(self)->bool:
		for directory in self.required_dirs:
			try:
				dir_path=os.path.join(self.base_dir,directory)
				if not os.path.exists(dir_path):
					os.makedirs(dir_path,exist_ok=True)
					if self.verbose:self.color.print(f"Created directory: {directory}",'green')
			except Exception as e:self.color.print(f"Error creating directory {directory}: {e}",'red');return False
		package_dirs=['maggie','maggie/core','maggie/extensions','maggie/utils','maggie/utils/config','maggie/utils/hardware','maggie/utils/llm','maggie/utils/stt','maggie/utils/tts']
		for pkg_dir in package_dirs:
			init_path=os.path.join(self.base_dir,pkg_dir,'__init__.py')
			if not os.path.exists(init_path):
				try:
					with open(init_path,'w')as f:f.write('# Maggie AI Assistant package\n')
					if self.verbose:self.color.print(f"Created __init__.py in {pkg_dir}",'green')
				except Exception as e:self.color.print(f"Error creating __init__.py in {pkg_dir}: {e}",'red');return False
		self.color.print('All required directories created successfully','green');return True
	def _setup_virtual_env(self)->bool:
		venv_dir=os.path.join(self.base_dir,'venv')
		if os.path.exists(venv_dir):self.color.print('Virtual environment already exists','yellow');return True
		python_cmd=sys.executable;returncode,_,stderr=self._run_command([python_cmd,'-m','venv',venv_dir])
		if returncode!=0:self.color.print(f"Error creating virtual environment: {stderr}",'red');return False
		self.color.print('Virtual environment created successfully','green');return True
	def _get_venv_python(self)->str:
		if self.platform_system=='Windows':return os.path.join(self.base_dir,'venv','Scripts','python.exe')
		else:return os.path.join(self.base_dir,'venv','bin','python')
	def _install_basic_dependencies(self,python_cmd:str)->bool:
		self.color.print('Upgrading pip, setuptools, and wheel...','cyan');returncode,_,stderr=self._run_command([python_cmd,'-m','pip','install','--upgrade','pip','setuptools','wheel'])
		if returncode!=0:self.color.print(f"Error upgrading pip, setuptools, and wheel: {stderr}",'red');return False
		basic_deps=['urllib3','tqdm','numpy','psutil','PyYAML','loguru','requests'];self.color.print(f"Installing basic packages: {', '.join(basic_deps)}...",'cyan');returncode,_,stderr=self._run_command([python_cmd,'-m','pip','install','--upgrade']+basic_deps)
		if returncode!=0:self.color.print(f"Error installing basic packages: {stderr}",'red');return False
		return True
	def _install_pytorch(self,python_cmd:str)->bool:
		if self.cpu_only:self.color.print('Installing PyTorch (CPU version)...','cyan');cmd=[python_cmd,'-m','pip','install','torch==2.0.1','torchvision==0.15.2','torchaudio==2.0.2']
		else:self.color.print('Installing PyTorch with CUDA 11.8 support (optimized for RTX 3080)...','cyan');cmd=[python_cmd,'-m','pip','install','torch==2.0.1+cu118','torchvision==0.15.2+cu118','torchaudio==2.0.2+cu118','--extra-index-url','https://download.pytorch.org/whl/cu118']
		returncode,_,stderr=self._run_command(cmd)
		if returncode!=0:self.color.print(f"Error installing PyTorch: {stderr}",'red');self.color.print('Continuing with installation, but GPU acceleration may not work','yellow');return False
		verify_cmd=[python_cmd,'-c',"import torch; print(f'PyTorch {torch.__version__} installed successfully'); print(f'CUDA available: {torch.cuda.is_available()}')"];returncode,stdout,_=self._run_command(verify_cmd,check=False)
		if returncode==0:
			for line in stdout.splitlines():self.color.print(line,'green')
			return True
		else:self.color.print('PyTorch installation verification failed','yellow');return False
	def _install_dependencies(self)->bool:
		python_cmd=self._get_venv_python()
		if not self._install_basic_dependencies(python_cmd):self.color.print('Failed to install basic dependencies','red');return False
		pytorch_success=self._install_pytorch(python_cmd)
		if not pytorch_success and not self.cpu_only:
			self.color.print('PyTorch with CUDA failed to install','yellow');response=self.color.input('Try installing CPU version instead? (y/n): ',color='magenta')
			if response.lower()=='y':cmd=[python_cmd,'-m','pip','install','torch==2.0.1','torchvision==0.15.2','torchaudio==2.0.2'];self._run_command(cmd)
		req_path=os.path.join(self.base_dir,'requirements.txt');temp_req_path=os.path.join(self.base_dir,'temp_requirements.txt')
		try:
			with open(req_path,'r')as f:req_content=f.read()
			filtered_lines=[]
			for line in req_content.splitlines():
				if not line or line.startswith('#')or'torch'in line or'cuda'in line or'PyAudio'in line or'whisper'in line.lower()or'kokoro'in line.lower():continue
				filtered_lines.append(line)
			with open(temp_req_path,'w')as f:f.write('\n'.join(filtered_lines))
			self.color.print('Installing standard dependencies...','cyan');returncode,_,stderr=self._run_command([python_cmd,'-m','pip','install','-r',temp_req_path]);os.remove(temp_req_path)
			if returncode!=0:self.color.print(f"Error installing standard dependencies: {stderr}",'red');self.color.print('Continuing with installation of critical components','yellow')
		except Exception as e:
			self.color.print(f"Error processing requirements file: {e}",'red')
			if os.path.exists(temp_req_path):os.remove(temp_req_path)
			return False
		self._install_specialized_dependencies(python_cmd);return True
	def _install_specialized_dependencies(self,python_cmd:str)->bool:
		self._install_pyaudio(python_cmd);self._install_kokoro(python_cmd)
		if not self._install_whisper(python_cmd):self.color.print('Warning: Failed to install faster-whisper','yellow');self.color.print('Speech recognition functionality may be limited','yellow');return False
		if not self.cpu_only:self.color.print('Installing GPU-specific dependencies...','cyan');self._run_command([python_cmd,'-m','pip','install','onnxruntime-gpu==1.15.1'])
		return True
	def _install_pyaudio(self,python_cmd:str)->bool:
		self.color.print('Installing PyAudio...','cyan');returncode,_,_=self._run_command([python_cmd,'-c',"import pyaudio; print('PyAudio already installed')"],check=False)
		if returncode==0:self.color.print('PyAudio already installed','green');return True
		if self.platform_system=='Windows':
			py_ver=f"{sys.version_info.major}{sys.version_info.minor}";wheel_url=f"https://files.pythonhosted.org/packages/27/bc/719d140ee63cf4b0725016531d36743a797ffdbab85e8536922902c9349a/PyAudio-0.2.14-cp310-cp310-win_amd64.whl";wheel_path=os.path.join(self.base_dir,'downloads','wheels','PyAudio-0.2.14-cp310-cp310-win_amd64.whl');os.makedirs(os.path.dirname(wheel_path),exist_ok=True)
			if not self._download_file(wheel_url,wheel_path):self.color.print('Failed to download PyAudio wheel','red');return False
			returncode,_,stderr=self._run_command([python_cmd,'-m','pip','install',wheel_path])
			if returncode==0:self.color.print('PyAudio installed successfully from wheel','green');return True
			else:
				self.color.print(f"Error installing PyAudio from wheel: {stderr}",'red');returncode,_,stderr=self._run_command([python_cmd,'-m','pip','install','PyAudio==0.2.13'])
				if returncode==0:self.color.print('PyAudio installed successfully','green');return True
				else:self.color.print(f"Failed to install PyAudio: {stderr}",'red');self.color.print('Audio input functionality will be limited','yellow');return False
		else:
			returncode,_,stderr=self._run_command([python_cmd,'-m','pip','install','PyAudio==0.2.13'])
			if returncode==0:self.color.print('PyAudio installed successfully','green');return True
			else:self.color.print(f"Error installing PyAudio: {stderr}",'red');self.color.print('You may need to install portaudio19-dev:','yellow');self.color.print('sudo apt-get install portaudio19-dev','yellow');self.color.print('Then try: pip install PyAudio==0.2.13','yellow');return False
	def _install_kokoro(self,python_cmd:str)->bool:
		self.color.print('Installing kokoro TTS engine...','cyan');returncode,_,_=self._run_command([python_cmd,'-c',"import kokoro; print('kokoro already installed')"],check=False)
		if returncode==0:self.color.print('kokoro already installed','green');return True
		self._run_command([python_cmd,'-m','pip','install','numpy','tqdm','soundfile'])
		if not self.cpu_only and self.hardware_info['gpu']['cuda_available']:self._run_command([python_cmd,'-m','pip','install','onnxruntime-gpu==1.15.1'])
		else:self._run_command([python_cmd,'-m','pip','install','onnxruntime==1.15.1'])
		if self.has_git:
			returncode,_,stderr=self._run_command([python_cmd,'-m','pip','install','git+https://github.com/hexgrad/kokoro.git'])
			if returncode==0:self.color.print('kokoro installed successfully from GitHub','green');return True
			else:self.color.print(f"Error installing kokoro from GitHub: {stderr}",'red');self.color.print('Trying alternative installation method...','yellow')
		kokoro_dir=os.path.join(self.base_dir,'downloads','kokoro');os.makedirs(kokoro_dir,exist_ok=True);zip_url='https://github.com/hexgrad/kokoro/archive/refs/heads/main.zip';zip_path=os.path.join(self.base_dir,'downloads','kokoro.zip')
		if not self._download_file(zip_url,zip_path):self.color.print('Failed to download kokoro repository','red');return False
		try:
			with zipfile.ZipFile(zip_path,'r')as zip_ref:zip_ref.extractall(os.path.dirname(kokoro_dir))
			extracted_dir=os.path.join(os.path.dirname(kokoro_dir),'kokoro-main')
			if os.path.exists(kokoro_dir):shutil.rmtree(kokoro_dir)
			shutil.move(extracted_dir,kokoro_dir);returncode,_,stderr=self._run_command([python_cmd,'-m','pip','install',kokoro_dir])
			if returncode==0:self.color.print('kokoro installed successfully from downloaded repository','green');return True
			else:
				self.color.print(f"Error installing kokoro from local directory: {stderr}",'red')
				if self.skip_problematic:self.color.print('Skipping kokoro installation (TTS may not work)','yellow');return False
				else:
					response=self.color.input('Failed to install kokoro. Skip this dependency? (y/n): ',color='magenta')
					if response.lower()=='y':self.color.print('Skipping kokoro installation','yellow');return False
					else:self.color.print('Installation cannot continue without kokoro','red');return False
		except Exception as e:self.color.print(f"Error extracting or installing kokoro: {e}",'red');return False
	def _install_whisper(self,python_cmd:str)->bool:
		self.color.print('Installing faster-whisper for speech recognition...','cyan');returncode,_,_=self._run_command([python_cmd,'-c',"import faster_whisper; print('faster-whisper already installed')"],check=False)
		if returncode==0:self.color.print('faster-whisper already installed','green');return True
		returncode,_,stderr=self._run_command([python_cmd,'-m','pip','install','faster-whisper==0.9.0','soundfile==0.12.1'])
		if returncode!=0:self.color.print(f"Error installing faster-whisper: {stderr}",'red');self.color.print('Speech recognition functionality may be limited','yellow');return False
		self.color.print('faster-whisper installed successfully','green');return True
	def _download_whisper_model(self)->bool:
		model_dir=os.path.join(self.base_dir,'maggie','models','stt','whisper-base.en');essential_files=['model.bin','config.json','tokenizer.json','vocab.json']
		if os.path.exists(model_dir):
			files_in_dir=os.listdir(model_dir)
			if all(essential_file in files_in_dir for essential_file in essential_files):self.color.print('Whisper base.en model is already available','green');return True
			else:self.color.print('Whisper model directory exists but appears incomplete','yellow');self.color.print('Will attempt to download/update the model','yellow')
		os.makedirs(model_dir,exist_ok=True)
		try:
			python_cmd=self._get_venv_python();self._run_command([python_cmd,'-m','pip','install','huggingface_hub']);check_script=f'''
import os
from huggingface_hub import HfApi

try:
    api = HfApi()
    # Check if the model repository exists and is accessible
    if api.repo_exists(repo_id="openai/whisper-base.en", repo_type="model"):
        print("Model repository is accessible")
    else:
        print("Model repository is not accessible")
        exit(1)
except Exception as e:
    print(f"Error checking model accessibility: {{e}}")
    exit(1)
''';check_code,check_stdout,check_stderr=self._run_command([python_cmd,'-c',check_script])
			if check_code!=0:self.color.print('Cannot access Whisper model repository','red');self.color.print(f"Error: {check_stderr}",'red');return False
			self.color.print('Downloading Whisper base.en model from Hugging Face (this may take a while)...','cyan');download_script=f'''
import os
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id="openai/whisper-base.en",
        local_dir="{model_dir.replace(os.sep,"/")}",
        ignore_patterns=["*.safetensors", "*.ot", "*.h5", "flax_model.*"],
        local_dir_use_symlinks=False
    )
    print("Model downloaded successfully")
except Exception as e:
    print(f"Error downloading model: {{e}}")
    exit(1)
''';returncode,stdout,stderr=self._run_command([python_cmd,'-c',download_script])
			if returncode!=0:self.color.print(f"Error downloading Whisper model: {stderr}",'red');return False
			files_in_dir=os.listdir(model_dir)
			if all(essential_file in files_in_dir for essential_file in essential_files):self.color.print('Whisper base.en model downloaded and verified successfully','green');return True
			else:self.color.print('Whisper model download appears incomplete','yellow');self.color.print('Missing files: '+', '.join([f for f in essential_files if f not in files_in_dir]),'yellow');return False
		except Exception as e:self.color.print(f"Error downloading Whisper model: {e}",'red');return False
	def _download_af_heart_model(self)->bool:
		model_dir=os.path.join(self.base_dir,'maggie','models','tts');model_path=os.path.join(model_dir,'af_heart.pt');MIN_SIZE=41*1024*1024
		if os.path.exists(model_path):
			file_size=os.path.getsize(model_path)
			if file_size>=MIN_SIZE:self.color.print(f"af_heart voice model verified ({file_size/(1024*1024):.2f} MB)",'green');return True
			else:self.color.print(f"af_heart model has incorrect size: {file_size/(1024*1024):.2f} MB",'yellow')
		os.makedirs(model_dir,exist_ok=True);model_urls=['https://huggingface.co/hexgrad/kokoro-voices/resolve/main/af_heart.pt','https://github.com/hexgrad/kokoro/releases/download/v0.1/af_heart.pt','https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.pt']
		for url in model_urls:
			self.color.print(f"Attempting download from: {url}",'cyan')
			try:
				with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'))as response:
					if response.getcode()!=200:self.color.print(f"URL inaccessible (status: {response.getcode()})",'yellow');continue
					content_length=response.getheader('Content-Length')
					if content_length and int(content_length)<MIN_SIZE:self.color.print(f"URL returns undersized file ({int(content_length)/(1024*1024):.2f} MB)",'yellow');continue
			except Exception as e:self.color.print(f"Error checking URL: {e}",'yellow');continue
			if self._download_file(url,model_path):
				file_size=os.path.getsize(model_path)
				if file_size>=MIN_SIZE:self.color.print(f"af_heart model download successful ({file_size/(1024*1024):.2f} MB)",'green');return True
				else:self.color.print(f"Downloaded file has incorrect size: {file_size/(1024*1024):.2f} MB",'yellow');os.remove(model_path)
		self.color.print('Failed to download af_heart voice model from any source','red');self.color.print('You may need to download it manually from: https://github.com/hexgrad/kokoro/releases','yellow');return False
	def _download_kokoro_onnx_models(self)->bool:
		model_dir=os.path.join(self.base_dir,'maggie','models','tts')
		os.makedirs(model_dir,exist_ok=True)
		
		# List of models to download with their URLs and minimum expected sizes
		models=[
			{
				'name':'kokoro-v1.0.onnx',
				'url':'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx',
				'min_size':10*1024*1024  # 10 MB minimum size check
			},
			{
				'name':'voices-v1.0.bin',
				'url':'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin',
				'min_size':5*1024*1024   # 5 MB minimum size check
			}
		]
		
		all_successful=True
		for model in models:
			model_path=os.path.join(model_dir,model['name'])
			
			# Check if file already exists and has sufficient size
			if os.path.exists(model_path):
				file_size=os.path.getsize(model_path)
				if file_size>=model['min_size']:
					self.color.print(f"{model['name']} already exists ({file_size/(1024*1024):.2f} MB) ✓",'green')
					continue
				else:
					self.color.print(f"{model['name']} has incorrect size: {file_size/(1024*1024):.2f} MB",'yellow')
			
			# Download the file
			self.color.print(f"Downloading {model['name']}...",'cyan')
			if self._download_file(model['url'],model_path):
				file_size=os.path.getsize(model_path)
				if file_size>=model['min_size']:
					self.color.print(f"{model['name']} download successful ({file_size/(1024*1024):.2f} MB)",'green')
				else:
					self.color.print(f"Downloaded file has incorrect size: {file_size/(1024*1024):.2f} MB",'yellow')
					os.remove(model_path)
					all_successful=False
			else:
				self.color.print(f"Failed to download {model['name']}",'red')
				all_successful=False
		
		if all_successful:
			self.color.print('All kokoro-onnx models downloaded successfully','green')
			return True
		else:
			self.color.print('Some kokoro-onnx models failed to download','yellow')
			return False
	def _download_mistral_model(self)->bool:
		if self.skip_models:self.color.print('Skipping Mistral model download (--skip-models)','yellow');return True
		mistral_dir=os.path.join(self.base_dir,'maggie','models','llm','mistral-7b-instruct-v0.3-GPTQ-4bit');essential_files=['config.json','tokenizer.json','tokenizer_config.json','quantize_config.json','special_tokens_map.json'];essential_patterns=[lambda files:any(file.endswith('.safetensors')for file in files)]
		if os.path.exists(mistral_dir)and os.listdir(mistral_dir):
			files_in_dir=os.listdir(mistral_dir);missing_files=[f for f in essential_files if f not in files_in_dir];failed_patterns=[i for(i,pattern_check)in enumerate(essential_patterns)if not pattern_check(files_in_dir)]
			if not missing_files and not failed_patterns:self.color.print('Mistral model is available and appears complete','green');return True
			else:
				self.color.print('Mistral model directory exists but appears incomplete','yellow')
				if missing_files:self.color.print(f"Missing files: {', '.join(missing_files)}",'yellow')
				if failed_patterns:self.color.print('Missing model weight files (.safetensors)','yellow')
				response=self.color.input('Would you like to try downloading the model again? (y/n): ',color='magenta')
				if response.lower()!='y':self.color.print('Continuing with existing model files','yellow');return True
		os.makedirs(mistral_dir,exist_ok=True)
		if not self.skip_models:
			response=self.color.input('Download Mistral 7B model? This requires ~5GB of storage (y/n): ',color='magenta')
			if response.lower()!='y':self.color.print('Skipping Mistral model download','yellow');return True
		if self.has_git:
			check_cmd=['git','ls-remote','https://huggingface.co/neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit'];returncode,_,stderr=self._run_command(check_cmd,check=False)
			if returncode!=0:self.color.print('Cannot access Mistral model repository','red');self.color.print(f"Error: {stderr}",'red');response=self.color.input('Continue installation without Mistral model? (y/n): ',color='magenta');return response.lower()=='y'
			self.color.print('Downloading Mistral 7B model using Git (this may take a while)...','cyan')
			if os.path.exists(mistral_dir)and os.listdir(mistral_dir):
				try:
					for item in os.listdir(mistral_dir):
						item_path=os.path.join(mistral_dir,item)
						if os.path.isfile(item_path):os.remove(item_path)
						elif os.path.isdir(item_path):shutil.rmtree(item_path)
				except Exception as e:self.color.print(f"Error cleaning existing model directory: {e}",'red');return False
			returncode,_,_=self._run_command(['git','clone','https://huggingface.co/neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit',mistral_dir],capture_output=False)
			if returncode==0:
				files_in_dir=os.listdir(mistral_dir);missing_files=[f for f in essential_files if f not in files_in_dir];failed_patterns=[i for(i,pattern_check)in enumerate(essential_patterns)if not pattern_check(files_in_dir)]
				if not missing_files and not failed_patterns:self.color.print('Mistral model downloaded and verified successfully','green');return True
				else:self.color.print('Mistral model download appears incomplete','yellow');response=self.color.input('Continue with potentially incomplete model? (y/n): ',color='magenta');return response.lower()=='y'
			else:self.color.print('Error downloading Mistral model with Git','red');self.color.print('LLM functionality will be limited','yellow');response=self.color.input('Continue installation without Mistral model? (y/n): ',color='magenta');return response.lower()=='y'
		else:self.color.print('Git not found, cannot download Mistral model','red');self.color.print('Install Git and rerun installation, or download model manually:','yellow');self.color.print('https://huggingface.co/neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit','yellow');response=self.color.input('Continue installation without Mistral model? (y/n): ',color='magenta');return response.lower()=='y'
	def _create_recipe_template(self)->bool:
		template_dir=os.path.join(self.base_dir,'maggie','templates');template_path=os.path.join(template_dir,'recipe_template.docx')
		if os.path.exists(template_path):self.color.print('Recipe template already exists','green');return True
		try:
			python_cmd=self._get_venv_python();returncode,_,_=self._run_command([python_cmd,'-c','\nimport docx\n\n# Create document\ndoc = docx.Document()\n\n# Add metadata section\ndoc.add_heading("Recipe Name", level=1)\n\n# Add metadata section\ndoc.add_heading("Recipe Information", level=2)\ninfo_table = doc.add_table(rows=3, cols=2)\ninfo_table.style = \'Table Grid\'\ninfo_table.cell(0, 0).text = "Preparation Time"\ninfo_table.cell(0, 1).text = "00 minutes"\ninfo_table.cell(1, 0).text = "Cooking Time"\ninfo_table.cell(1, 1).text = "00 minutes"\ninfo_table.cell(2, 0).text = "Servings"\ninfo_table.cell(2, 1).text = "0 servings"\n\n# Add ingredients section\ndoc.add_heading("Ingredients", level=2)\ndoc.add_paragraph("• Ingredient 1", style=\'ListBullet\')\ndoc.add_paragraph("• Ingredient 2", style=\'ListBullet\')\ndoc.add_paragraph("• Ingredient 3", style=\'ListBullet\')\n\n# Add steps section\ndoc.add_heading("Instructions", level=2)\ndoc.add_paragraph("1. Step 1", style=\'ListNumber\')\ndoc.add_paragraph("2. Step 2", style=\'ListNumber\')\ndoc.add_paragraph("3. Step 3", style=\'ListNumber\')\n\n# Add notes section\ndoc.add_heading("Notes", level=2)\ndoc.add_paragraph("Add any additional notes, tips, or variations here.")\n\n# Save template\ndoc.save("{}")\n                '.format(template_path.replace('\\','\\\\'))])
			if returncode==0:self.color.print('Recipe template created successfully','green');return True
			else:
				returncode,_,_=self._run_command([python_cmd,'main.py','--create-template'])
				if returncode==0:self.color.print('Recipe template created with main.py','green');return True
				else:self.color.print('Failed to create recipe template','red');self.color.print('Recipe creator extension may not work properly','yellow');return False
		except Exception as e:self.color.print(f"Error creating recipe template: {e}",'red');return False
	def _setup_config(self)->bool:
		config_path=os.path.join(self.base_dir,'config.yaml');example_path=os.path.join(self.base_dir,'config.yaml.example');alt_example_path=os.path.join(self.base_dir,'config-yaml-example.txt')
		if not os.path.exists(example_path)and os.path.exists(alt_example_path):
			try:shutil.copy(alt_example_path,example_path);self.color.print(f"Created config example from {alt_example_path}",'green')
			except Exception as e:self.color.print(f"Error creating config example: {e}",'red');return False
		import json;temp_hardware_file=os.path.join(self.base_dir,'hardware_info.json')
		try:
			with open(temp_hardware_file,'w')as f:json.dump(self.hardware_info,f)
		except Exception as e:self.color.print(f"Error writing hardware info: {e}",'red');return False
		python_cmd=self._get_venv_python();normalized_base_dir=os.path.normpath(str(self.base_dir));base_dir_str=repr(normalized_base_dir);code=f"""import yaml
import json
import os

base_dir = {base_dir_str}
config_path = os.path.join(base_dir, 'config.yaml')
example_path = os.path.join(base_dir, 'config.yaml.example')
hardware_file = os.path.join(base_dir, 'hardware_info.json')

# Load hardware info
with open(hardware_file, 'r') as f:
    hardware_info = json.load(f)

# Check if config exists
if not os.path.exists(config_path):
    if not os.path.exists(example_path):
        print(\"Error: Configuration example file not found\")
        exit(1)
    # Copy example to config
    with open(example_path, 'r') as f:
        config = yaml.safe_load(f)
else:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

# Set TTS voice model
if 'tts' in config and 'voice_model' in config['tts']:
    config['tts']['voice_model'] = 'af_heart.pt'

# Set STT model path for Whisper
if 'speech' in config and 'whisper' in config['speech']:
    config['speech']['whisper']['model_path'] = 'models/stt/whisper-base.en'

# Optimize for hardware
if hardware_info['gpu']['is_rtx_3080']:
    if 'llm' in config:
        config['llm']['gpu_layers'] = 32
        config['llm']['gpu_layer_auto_adjust'] = True
    if 'gpu' not in config:
        config['gpu'] = {{}}
    config['gpu']['max_percent'] = 90
    config['gpu']['model_unload_threshold'] = 95
    if 'speech' in config and 'whisper' in config['speech']:
        config['speech']['whisper']['compute_type'] = 'float16'
    if 'tts' in config:
        config['tts']['gpu_acceleration'] = True
        config['tts']['gpu_precision'] = 'mixed_float16'
elif {self.cpu_only}:
    if 'llm' in config:
        config['llm']['gpu_layers'] = 0
        config['llm']['gpu_layer_auto_adjust'] = False
    if 'gpu' not in config:
        config['gpu'] = {{}}
    config['gpu']['max_percent'] = 0
    config['gpu']['model_unload_threshold'] = 0
    if 'tts' in config:
        config['tts']['gpu_acceleration'] = False

if hardware_info['cpu']['is_ryzen_9_5900x']:
    if 'cpu' not in config:
        config['cpu'] = {{}}
    config['cpu']['max_threads'] = 8
    config['cpu']['thread_timeout'] = 30

if hardware_info['memory']['is_32gb']:
    if 'memory' not in config:
        config['memory'] = {{}}
    config['memory']['max_percent'] = 75
    config['memory']['model_unload_threshold'] = 85

# Write config
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# Clean up
os.remove(hardware_file)
""";returncode,_,stderr=self._run_command([python_cmd,'-c',code],cwd=self.base_dir)
		if returncode!=0:self.color.print(f"Error setting up configuration: {stderr}",'red');return False
		self.color.print('Configuration file created with optimized settings','green');self.color.print('NOTE: You must edit config.yaml to add your Picovoice access key','yellow');return True
	def _optimize_config_for_hardware(self,config:Dict[str,Any])->None:
		if not self.cpu_only and self.hardware_info['gpu']['is_rtx_3080']:
			if'llm'in config:config['llm']['gpu_layers']=32;config['llm']['gpu_layer_auto_adjust']=True
			if'gpu'not in config:config['gpu']={}
			config['gpu']['max_percent']=90;config['gpu']['model_unload_threshold']=95
			if'speech'in config and'whisper'in config['speech']:config['speech']['whisper']['compute_type']='float16'
			if'tts'in config:config['tts']['gpu_acceleration']=True;config['tts']['gpu_precision']='mixed_float16'
		elif self.cpu_only:
			if'llm'in config:config['llm']['gpu_layers']=0;config['llm']['gpu_layer_auto_adjust']=False
			if'gpu'not in config:config['gpu']={}
			config['gpu']['max_percent']=0;config['gpu']['model_unload_threshold']=0
			if'tts'in config:config['tts']['gpu_acceleration']=False
		if self.hardware_info['cpu']['is_ryzen_9_5900x']:
			if'cpu'not in config:config['cpu']={}
			config['cpu']['max_threads']=8;config['cpu']['thread_timeout']=30
		if self.hardware_info['memory']['is_32gb']:
			if'memory'not in config:config['memory']={}
			config['memory']['max_percent']=75;config['memory']['model_unload_threshold']=85
	def _install_extensions(self)->bool:
		python_cmd=self._get_venv_python();self.color.print('Installing recipe_creator extension dependencies...','cyan');returncode,_,stderr=self._run_command([python_cmd,'-m','pip','install','python-docx>=0.8.11'])
		if returncode!=0:self.color.print(f"Error installing recipe_creator dependencies: {stderr}",'red');self.color.print('Recipe creator extension may not work properly','yellow');return False
		self.color.print('Extensions installed successfully','green');return True
	def verify_system(self)->bool:
		self.progress.start_step('Verifying system compatibility');python_compatible=self._verify_python_version()
		if not python_compatible:self.progress.complete_step(False,'Incompatible Python version');return False
		self.hardware_info=self._detect_hardware();self._check_tools();self.progress.complete_step(True);return True
	def install(self)->bool:
		self.color.print('\n=== Maggie AI Assistant Installation ===','cyan',bold=True);self.color.print(f"Platform: {self.platform_system} ({platform.platform()})",'cyan');self.color.print(f"Python: {platform.python_version()}",'cyan')
		if self.cpu_only:self.color.print('Mode: CPU-only (no GPU acceleration)','yellow')
		if not self.verify_system():return False
		self.progress.start_step('Creating directory structure')
		if not self._create_directories():self.progress.complete_step(False,'Failed to create directories');return False
		self.progress.complete_step(True);self.progress.start_step('Setting up virtual environment')
		if not self._setup_virtual_env():self.progress.complete_step(False,'Failed to create virtual environment');return False
		self.progress.complete_step(True);self.progress.start_step('Installing dependencies')
		if not self._install_dependencies():
			self.progress.complete_step(False,'Some dependencies failed to install')
			if not self.color.input('Continue with installation? (y/n): ',color='magenta').lower()=='y':return False
		else:self.progress.complete_step(True)
		self.progress.start_step('Setting up configuration')
		if not self._setup_config():self.progress.complete_step(False,'Failed to set up configuration');return False
		self.progress.complete_step(True);self.progress.start_step('Downloading models')
		if not self._download_af_heart_model():self.color.print('Warning: Failed to download TTS voice model','yellow');self.color.print('Text-to-speech functionality may be limited','yellow')
		if not self._download_kokoro_onnx_models():self.color.print('Warning: Failed to download some kokoro-onnx model files','yellow');self.color.print('ONNX-based TTS functionality may be limited','yellow')
		if not self._download_whisper_model():self.color.print('Warning: Failed to download Whisper model','yellow');self.color.print('Speech recognition functionality may be limited','yellow')
		if not self.skip_models:self._download_mistral_model()
		else:self.color.print('Skipping LLM model download (--skip-models)','yellow')
		self.progress.complete_step(True);self.progress.start_step('Setting up extensions')
		if not self._create_recipe_template():self.color.print('Warning: Failed to create recipe template','yellow')
		if not self._install_extensions():self.color.print('Warning: Some extensions may not work properly','yellow')
		self.progress.complete_step(True);self.progress.start_step('Completing installation');self.progress.display_summary(True);self.color.print('\nTo start Maggie AI Assistant:','cyan',bold=True)
		if self.platform_system=='Windows':self.color.print('   .\\venv\\Scripts\\activate','green');self.color.print('   python main.py','green')
		else:self.color.print('   source venv/bin/activate','green');self.color.print('   python main.py','green')
		self.color.print('\nImportant Notes:','cyan',bold=True);self.color.print('1. Edit config.yaml to add your Picovoice access key','yellow');self.color.print('   Visit https://console.picovoice.ai/ to obtain a key','yellow')
		if not self.has_git:self.color.print('2. Git is not installed - some features may be limited','yellow');self.color.print('   Install Git for full functionality','yellow')
		self.color.print('3. The Whisper speech recognition model is now included in the project','green');self.color.print('   No need to install whisper_streaming separately','green');self.progress.complete_step(True);response=self.color.input('\nWould you like to start Maggie now? (y/n): ',color='magenta')
		if response.lower()=='y':self.color.print('\nStarting Maggie AI Assistant...','cyan',bold=True);python_cmd=self._get_venv_python();self._run_command([python_cmd,'main.py'],capture_output=False)
		return True
def main()->int:
	parser=argparse.ArgumentParser(description='Maggie AI Assistant Installer',formatter_class=argparse.ArgumentDefaultsHelpFormatter);parser.add_argument('--verbose',action='store_true',help='Enable verbose output');parser.add_argument('--cpu-only',action='store_true',help='Install CPU-only version (no GPU acceleration)');parser.add_argument('--skip-models',action='store_true',help='Skip downloading large LLM models');parser.add_argument('--skip-problematic',action='store_true',help='Skip problematic dependencies that may cause installation issues');parser.add_argument('--force-reinstall',action='store_true',help='Force reinstallation of already installed packages');args=parser.parse_args();installer=MaggieInstaller(verbose=args.verbose,cpu_only=args.cpu_only,skip_models=args.skip_models,skip_problematic=args.skip_problematic,force_reinstall=args.force_reinstall)
	try:success=installer.install();return 0 if success else 1
	except KeyboardInterrupt:print('\nInstallation cancelled by user');return 1
	except Exception as e:print(f"\nUnexpected error during installation: {e}");return 1
if __name__=='__main__':sys.exit(main())