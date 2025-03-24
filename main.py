#!/usr/bin/env python3
import os,argparse,sys,platform,multiprocessing,time,yaml
from typing import Dict,Any,Optional,List,Tuple
from loguru import logger
__all__=['main','parse_arguments','setup_logging','verify_system']
def error_publisher(message):
	if hasattr(error_publisher,'event_bus'):error_publisher.event_bus.publish('error_logged',message)
def create_event_bus_handler(event_bus):
	def handler(record):
		if record['level'].name=='ERROR'or record['level'].name=='CRITICAL':error_data={'message':record['message'],'source':record['name'],'file':record['file'].name if hasattr(record['file'],'name')else str(record['file']),'line':record['line'],'level':record['level'].name};event_bus.publish('error_logged',error_data)
	return handler
def parse_arguments()->argparse.Namespace:parser=argparse.ArgumentParser(description='Maggie AI Assistant',formatter_class=argparse.ArgumentDefaultsHelpFormatter);parser.add_argument('--config',type=str,default='config.yaml',help='Path to configuration file');parser.add_argument('--debug',action='store_true',help='Enable debug logging');parser.add_argument('--verify',action='store_true',help='Verify system configuration without starting the assistant');parser.add_argument('--create-template',action='store_true',help="Create the recipe template file if it doesn't exist");parser.add_argument('--optimize',action='store_true',help='Optimize configuration for detected hardware');parser.add_argument('--headless',action='store_true',help='Run in headless mode without GUI');return parser.parse_args()
def setup_logging(debug:bool=False)->None:os.makedirs('logs',exist_ok=True);log_level='DEBUG'if debug else'INFO';logger.configure(handlers=[{'sink':sys.stdout,'level':log_level,'format':'<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'},{'sink':'logs/maggie.log','rotation':'10 MB','retention':'1 week','format':'{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}'}]);log_system_info()
def log_system_info()->None:
	try:logger.info(f"System: {platform.system()} {platform.release()}");logger.info(f"Python: {platform.python_version()}");log_cpu_info();log_ram_info();log_gpu_info()
	except ImportError as e:logger.warning(f"System info modules not available: {e}")
	except Exception as e:logger.warning(f"Error gathering system information: {e}")
def log_cpu_info()->None:
	try:import psutil;cpu_info=platform.processor();cpu_cores=psutil.cpu_count(logical=False);cpu_threads=psutil.cpu_count(logical=True);logger.info(f"CPU: {cpu_info}");logger.info(f"CPU Cores: {cpu_cores} physical, {cpu_threads} logical")
	except ImportError:logger.warning('psutil not available for CPU information')
	except Exception as e:logger.warning(f"Error getting CPU information: {e}")
def log_ram_info()->None:
	try:import psutil;ram_gb=psutil.virtual_memory().total/1024**3;logger.info(f"RAM: {ram_gb:.2f} GB")
	except ImportError:logger.warning('psutil not available for RAM information')
	except Exception as e:logger.warning(f"Error getting RAM information: {e}")
def log_gpu_info()->None:
	try:
		import torch
		if torch.cuda.is_available():
			for i in range(torch.cuda.device_count()):
				gpu_name=torch.cuda.get_device_name(i);memory_gb=torch.cuda.get_device_properties(i).total_memory/1024**3;logger.info(f"GPU {i}: {gpu_name}");logger.info(f"GPU Memory: {memory_gb:.2f} GB")
				if'3080'in gpu_name:logger.info(f"RTX 3080 detected - Tensor Cores available");logger.info(f"Optimizing for 10GB VRAM and Ampere architecture")
		else:logger.warning('CUDA not available, GPU acceleration disabled')
	except ImportError:logger.warning('PyTorch not installed, GPU detection skipped')
	except Exception as e:logger.warning(f"Error detecting GPU information: {e}")
def verify_system()->bool:
	verification_issues=[];logger.info('Verifying system configuration...');python_version_valid=check_python_version()
	if not python_version_valid:verification_issues.append('Incompatible Python version')
	gpu_valid,gpu_warnings=check_gpu_compatibility()
	if not gpu_valid:verification_issues.append('GPU compatibility issues detected')
	dirs_valid=check_required_directories()
	if not dirs_valid:verification_issues.append('Directory issues detected')
	deps_valid=check_dependencies()
	if not deps_valid:verification_issues.append('Missing dependencies')
	memory_valid=check_memory_configuration()
	if not memory_valid:verification_issues.append('Insufficient memory configuration')
	if verification_issues:
		logger.error('System verification failed with the following issues:')
		for issue in verification_issues:logger.error(f"  - {issue}")
		return False
	logger.info('System verification completed successfully');return True
def check_gpu_compatibility()->Tuple[bool,List[str]]:
	warnings=[]
	try:
		import torch
		if torch.cuda.is_available():
			device_name=torch.cuda.get_device_name(0);logger.info(f"CUDA available: {torch.version.cuda}");logger.info(f"GPU detected: {device_name}");gpu_memory=torch.cuda.get_device_properties(0).total_memory/1024**3;logger.info(f"VRAM: {gpu_memory:.2f}GB");is_rtx_3080='3080'in device_name
			if is_rtx_3080:
				logger.info('RTX 3080 detected - optimal hardware configuration')
				if gpu_memory<9.5:warnings.append(f"RTX 3080 VRAM ({gpu_memory:.1f}GB) is less than expected 10GB");logger.warning(f"RTX 3080 VRAM ({gpu_memory:.1f}GB) is less than expected 10GB")
				compute_capability=torch.cuda.get_device_capability(0);cc_version=f"{compute_capability[0]}.{compute_capability[1]}"
				if cc_version!='8.6':warnings.append(f"RTX 3080 compute capability ({cc_version}) is not 8.6");logger.warning(f"RTX 3080 compute capability ({cc_version}) is not 8.6")
				cuda_version=torch.version.cuda
				if not cuda_version.startswith('11.'):warnings.append(f"CUDA version {cuda_version} - version 11.x recommended for RTX 3080");logger.warning(f"CUDA version {cuda_version} - version 11.x recommended for RTX 3080")
				if hasattr(torch.version,'cuda_compiled_version')and torch.version.cuda!=torch.version.cuda_compiled_version:warnings.append(f"CUDA runtime version ({torch.version.cuda}) differs from compiled version ({torch.version.cuda_compiled_version})");logger.warning(f"CUDA runtime version ({torch.version.cuda}) differs from compiled version ({torch.version.cuda_compiled_version})")
				available_vram=(torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_allocated()-torch.cuda.memory_reserved())/1024**3
				if available_vram<8.:warnings.append(f"Only {available_vram:.1f}GB VRAM available - less than 8GB may cause issues with large models");logger.warning(f"Only {available_vram:.1f}GB VRAM available - less than 8GB may cause issues with large models")
			elif gpu_memory<8:warnings.append(f"GPU memory ({gpu_memory:.1f}GB) is less than recommended 8GB");logger.warning(f"GPU memory ({gpu_memory:.1f}GB) is less than recommended 8GB")
			try:test_tensor=torch.ones(1000,1000,device='cuda');test_result=torch.matmul(test_tensor,test_tensor);del test_tensor,test_result;torch.cuda.empty_cache();logger.info('CUDA operations test successful')
			except Exception as e:warnings.append(f"CUDA operations test failed: {e}");logger.error(f"CUDA operations test failed: {e}");return False,warnings
			return True,warnings
		else:logger.warning('CUDA not available, GPU acceleration disabled');logger.warning('Performance may be significantly reduced without GPU acceleration');warnings.append('CUDA not available - running in CPU-only mode will be slow');return True,warnings
	except ImportError:logger.error('PyTorch not installed, GPU acceleration unavailable');logger.error('Install PyTorch with: pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118');warnings.append('PyTorch not available - GPU acceleration disabled');return True,warnings
	except Exception as e:logger.error(f"Error checking GPU compatibility: {e}");warnings.append(f"Error checking GPU: {e}");return True,warnings
def check_memory_configuration()->bool:
	try:
		import psutil;memory=psutil.virtual_memory();total_gb=memory.total/1024**3;available_gb=memory.available/1024**3;logger.info(f"Total RAM: {total_gb:.2f}GB, Available: {available_gb:.2f}GB")
		if total_gb<8:logger.error(f"Critical: Insufficient RAM: {total_gb:.2f}GB (minimum 8GB required)");return False
		if total_gb<16:logger.warning(f"Insufficient RAM: {total_gb:.2f}GB (minimum 16GB recommended)")
		elif total_gb<32:logger.warning(f"RAM: {total_gb:.2f}GB (32GB recommended for optimal performance)")
		else:logger.info(f"RAM: {total_gb:.2f}GB (optimal)")
		if available_gb<2:logger.error(f"Critical: Only {available_gb:.2f}GB RAM available. Close other applications.");return False
		return True
	except ImportError:logger.warning('psutil not available for memory checks');logger.warning('Install psutil for comprehensive memory verification');return True
	except Exception as e:logger.error(f"Error checking memory configuration: {e}");logger.warning('Memory check failed, proceeding with caution');return True
def check_python_version()->bool:
	python_version=platform.python_version_tuple()
	if int(python_version[0])!=3 or int(python_version[1])!=10:error_msg=f"Unsupported Python version: {platform.python_version()}";logger.error(error_msg);logger.error('Maggie requires Python 3.10.x specifically. Other versions are not compatible.');logger.error('Please install Python 3.10 and try again.');return False
	else:logger.info(f"Python version {platform.python_version()} is compatible");return True
def check_cuda_availability()->bool:
	try:
		import torch
		if torch.cuda.is_available():
			logger.info(f"CUDA available: {torch.version.cuda}");logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}");is_rtx_3080='3080'in torch.cuda.get_device_name(0)
			if is_rtx_3080:logger.info('RTX 3080 detected - optimal hardware configuration')
			else:
				gpu_memory=torch.cuda.get_device_properties(0).total_memory/1024**3
				if gpu_memory<8:logger.warning(f"GPU memory ({gpu_memory:.1f}GB) is less than recommended 10GB");logger.warning('Performance may be degraded')
		else:logger.warning('CUDA not available, GPU acceleration disabled');logger.warning('Performance may be significantly reduced without GPU acceleration')
		return True
	except ImportError:logger.error('PyTorch not installed, GPU acceleration unavailable');logger.error('Install PyTorch with: pip install torch');return True
	except Exception as e:logger.error(f"Error checking CUDA: {e}");return True
def check_required_directories()->bool:
	required_dirs=['maggie/models','maggie/models/tts','logs','maggie/extensions/recipe_creator/recipes','maggie/extensions/recipe_creator/templates']
	for directory in required_dirs:
		if not os.path.exists(directory):
			try:os.makedirs(directory,exist_ok=True);logger.info(f"Created directory: {directory}")
			except Exception as e:error_msg=f"Failed to create directory {directory}: {e}";logger.error(error_msg);return False
	return True
def check_dependencies()->bool:
	pyaudio_installed=check_pyaudio();critical_deps=['pvporcupine','faster_whisper','ctransformers','transitions','docx','PySide6'];missing_deps=[]
	for dep in critical_deps:
		try:
			module_name=dep.replace('-','_')
			if dep=='PySide6':module_name='PySide6.QtCore'
			__import__(module_name)
		except ImportError:missing_deps.append(dep)
	if missing_deps:deps_str=', '.join(missing_deps);logger.error(f"Missing critical dependencies: {deps_str}");logger.error('Install with: pip install -r requirements.txt');return False
	return True
def check_pyaudio()->bool:
	try:import pyaudio;logger.info('PyAudio found');return True
	except ImportError:
		error_msg='PyAudio not installed';logger.error(error_msg)
		if platform.system()=='Windows':logger.error('On Windows, install with: pip install PyAudio')
		else:logger.error('On Linux, first install portaudio19-dev, then install pyaudio')
		return False
def create_recipe_template()->bool:
	template_path='templates/recipe_template.docx'
	if os.path.exists(template_path):logger.info(f"Recipe template already exists at {template_path}");return True
	try:from docx import Document;os.makedirs('templates',exist_ok=True);doc=Document();doc.add_heading('Recipe Name',level=1);doc.add_heading('Recipe Information',level=2);info_table=doc.add_table(rows=3,cols=2);info_table.style='Table Grid';info_table.cell(0,0).text='Preparation Time';info_table.cell(0,1).text='00 minutes';info_table.cell(1,0).text='Cooking Time';info_table.cell(1,1).text='00 minutes';info_table.cell(2,0).text='Servings';info_table.cell(2,1).text='0 servings';doc.add_heading('Ingredients',level=2);doc.add_paragraph('• Ingredient 1',style='ListBullet');doc.add_paragraph('• Ingredient 2',style='ListBullet');doc.add_paragraph('• Ingredient 3',style='ListBullet');doc.add_heading('Instructions',level=2);doc.add_paragraph('1. Step 1',style='ListNumber');doc.add_paragraph('2. Step 2',style='ListNumber');doc.add_paragraph('3. Step 3',style='ListNumber');doc.add_heading('Notes',level=2);doc.add_paragraph('Add any additional notes, tips, or variations here.');doc.add_heading('Nutrition Information (per serving)',level=2);doc.add_paragraph('Calories: 000');doc.add_paragraph('Protein: 00g');doc.add_paragraph('Carbohydrates: 00g');doc.add_paragraph('Fat: 00g');doc.save(template_path);logger.info(f"Created recipe template at {template_path}");return True
	except ImportError:logger.error('python-docx not installed, cannot create template');return False
	except Exception as e:logger.error(f"Failed to create recipe template: {e}");return False
def optimize_system()->bool:
	try:
		if platform.system()=='Windows':windows_optimizations()
		elif platform.system()=='Linux':linux_optimizations()
		common_optimizations();return True
	except Exception as e:logger.error(f"Failed to optimize system: {e}");return False
def windows_optimizations()->None:
	try:
		import psutil;process=psutil.Process(os.getpid());process.nice(psutil.HIGH_PRIORITY_CLASS);logger.info('Set process priority to high');cpu_count=psutil.cpu_count(logical=True)
		if cpu_count>=16:affinity=list(range(16));process.cpu_affinity(affinity);logger.info(f"Set CPU affinity to first 8 physical cores: {affinity}")
	except ImportError:logger.warning('psutil not available for Windows optimizations')
	except Exception as e:logger.warning(f"Failed to apply Windows performance optimizations: {e}")
def linux_optimizations()->None:
	try:
		is_root=os.geteuid()==0
		if is_root:os.system('echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor');logger.info('Set CPU governor to performance mode')
		else:logger.warning('Not running as root, skipping some system optimizations')
		os.nice(-10);logger.info('Set process nice level to -10')
	except Exception as e:logger.warning(f"Failed to apply Linux performance optimizations: {e}")
def common_optimizations()->None:
	try:import setproctitle;setproctitle.setproctitle('maggie-ai-assistant');logger.info("Set process title to 'maggie-ai-assistant'")
	except ImportError:pass
	if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(4096)
def main()->int:
	args=parse_arguments();os.makedirs('logs',exist_ok=True);setup_logging(args.debug);logger.info('Starting Maggie AI Assistant');logger.info(f"Running on Python {platform.python_version()}");logger.info(f"Process ID: {os.getpid()}");setup_multiprocessing()
	if args.optimize:
		if optimize_system():logger.info('System optimized for performance')
		else:logger.warning('System optimization failed')
	if args.create_template:
		create_recipe_template()
		if not args.verify:return 0
	if args.verify:
		if verify_system():logger.info('System verification successful');return 0
		else:logger.error('System verification failed');return 1
	if not verify_system():logger.warning('System verification failed, but attempting to start anyway')
	return start_maggie(args)
def setup_multiprocessing()->None:
	try:multiprocessing.set_start_method('spawn');logger.info("Set multiprocessing start method to 'spawn'")
	except RuntimeError:pass
def start_maggie(args:argparse.Namespace)->int:
	try:
		if not args.headless:
			add_pyside6_paths()
			try:from PySide6.QtWidgets import QApplication;from maggie.utils.gui import MainWindow
			except ImportError as e:logger.error(f"GUI components import failed: {e}");logger.info('Falling back to headless mode');args.headless=True
		from maggie.core import MaggieAI,State
	except ImportError as e:logger.error(f"Failed to import required module: {e}");return 1
	config={}
	if os.path.exists(args.config):
		try:
			with open(args.config,'r')as f:
				config=yaml.safe_load(f)
				if config is None:logger.warning(f"Empty config file: {args.config}, using defaults");config={}
		except Exception as e:logger.error(f"Error reading config file: {e}")
	else:logger.warning(f"Config file not found: {args.config}, using defaults")
	maggie=MaggieAI(config);register_signal_handlers(maggie);error_publisher.event_bus=maggie.event_bus;logger.add(error_publisher,format='[{time:%H:%M:%S}] {level.name}: {message}',level='ERROR');success=maggie.start()
	if not success:logger.error('Failed to start Maggie AI core services');return 1
	if not args.headless:
		try:app=QApplication(sys.argv);window=MainWindow(maggie);maggie.set_gui(window);window.show();return app.exec()
		except Exception as e:logger.error(f"Error starting GUI: {e}");maggie.shutdown();return 1
	elif args.headless:
		logger.info('Running in headless mode')
		try:
			while maggie.state!=State.SHUTDOWN:time.sleep(1)
			return 0
		except KeyboardInterrupt:logger.info('Keyboard interrupt received, shutting down');maggie.shutdown();return 0
def register_signal_handlers(maggie)->None:
	try:
		import signal
		def signal_handler(sig,frame):logger.info(f"Received signal {sig}, shutting down gracefully");maggie.shutdown();sys.exit(0)
		signal.signal(signal.SIGINT,signal_handler);signal.signal(signal.SIGTERM,signal_handler);logger.info('Registered signal handlers for graceful shutdown')
	except Exception as e:logger.warning(f"Failed to register signal handlers: {e}")
def add_pyside6_paths():
	try:
		import subprocess,json,re;pip_show_pyside6=subprocess.run([sys.executable,'-m','pip','show','pyside6'],capture_output=True,text=True).stdout.strip();fp=re.search('location:\\s*(.+)',pip_show_pyside6.lower());bp=None if not fp else fp.group(1).strip();pyside6_paths=[]if not bp else[os.path.join(bp,'PySide6'),os.path.join(bp,'PySide6','Qt6'),os.path.join(bp,'PySide6','Qt6','bin')];result=[p for p in pyside6_paths if os.path.exists(p)]
		if result and len(result)>0:
			for p in result:
				if p not in sys.path:sys.path.append(p)
	except Exception as e:print(f"Error finding PySide6 paths using pip: {e}")
	try:
		import site
		for site_dir in site.getsitepackages():
			pyside_dir=os.path.join(site_dir,'PySide6')
			if os.path.exists(pyside_dir):
				pyside_paths=[pyside_dir,os.path.join(pyside_dir,'Qt6'),os.path.join(pyside_dir,'Qt6','bin')];result=[p for p in pyside_paths if os.path.exists(p)]
				if result and len(result)>0:
					for p in result:
						if p not in sys.path:sys.path.append(p)
	except Exception as e:print(f"Error finding PySide6 paths using site-packages: {e}")
	try:
		venv_dir=os.path.dirname(os.path.dirname(sys.executable))
		if os.path.exists(venv_dir):
			potential_paths=[os.path.join(venv_dir,'Lib','site-packages','PySide6'),os.path.join(venv_dir,'lib','python3.10','site-packages','PySide6'),os.path.join(venv_dir,'lib','site-packages','PySide6')]
			for pp in potential_paths:
				if os.path.exists(pp):
					pyside_paths=[pp,os.path.join(pp,'Qt6'),os.path.join(pp,'Qt6','bin')];result=[p for p in pyside_paths if os.path.exists(p)]
					if result and len(result)>0:
						for p in result:
							if p not in sys.path:sys.path.append(p)
	except Exception as e:print(f"Error finding PySide6 in virtual environment: {e}")
	return
if __name__=='__main__':sys.exit(main())