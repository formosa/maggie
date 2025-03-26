import os,sys,argparse,signal,platform,multiprocessing,time
from typing import Dict,Any,Optional,Tuple
from maggie.core import MaggieAI,State
from maggie.utils.config.manager import ConfigManager
from maggie.utils.logging import ComponentLogger,LoggingManager
from maggie.utils.error_handling import safe_execute,ErrorCategory,ErrorSeverity,record_error,with_error_handling
from maggie.utils.resource.detector import HardwareDetector
logger=ComponentLogger('Main')
def parse_arguments()->argparse.Namespace:parser=argparse.ArgumentParser(description='Maggie AI Assistant',formatter_class=argparse.ArgumentDefaultsHelpFormatter);parser.add_argument('--config',type=str,default='config.yaml',help='Path to configuration file');parser.add_argument('--debug',action='store_true',help='Enable debug logging');parser.add_argument('--verify',action='store_true',help='Verify system configuration without starting the assistant');parser.add_argument('--create-template',action='store_true',help="Create the recipe template file if it doesn't exist");parser.add_argument('--headless',action='store_true',help='Run in headless mode without GUI');return parser.parse_args()
def initialize_logging(config:Dict[str,Any],debug:bool=False)->None:
	os.makedirs('logs',exist_ok=True)
	if debug and'logging'in config:config['logging']['console_level']='DEBUG';config['logging']['file_level']='DEBUG'
	LoggingManager.initialize(config);LoggingManager.get_instance().setup_global_exception_handler();logger.info('Logging system initialized')
def initialize_multiprocessing()->None:
	try:multiprocessing.set_start_method('spawn');logger.info("Set multiprocessing start method to 'spawn'")
	except RuntimeError:pass
def setup_pyside6_paths()->None:
	try:
		import site
		for site_path in site.getsitepackages():
			pyside_path=os.path.join(site_path,'PySide6')
			if os.path.exists(pyside_path):
				if pyside_path not in sys.path:
					sys.path.append(pyside_path);qt6_path=os.path.join(pyside_path,'Qt6')
					if os.path.exists(qt6_path)and qt6_path not in sys.path:sys.path.append(qt6_path)
					bin_path=os.path.join(qt6_path,'bin')
					if os.path.exists(bin_path)and bin_path not in sys.path:sys.path.append(bin_path)
					break
	except Exception as e:logger.warning(f"Error setting up PySide6 paths: {e}")
def register_signal_handlers(maggie:MaggieAI)->None:
	try:
		def signal_handler(sig,frame):logger.info(f"Received signal {sig}, shutting down gracefully");maggie.shutdown();sys.exit(0)
		signal.signal(signal.SIGINT,signal_handler);signal.signal(signal.SIGTERM,signal_handler);logger.info('Registered signal handlers for graceful shutdown')
	except Exception as e:logger.warning(f"Failed to register signal handlers: {e}")
@with_error_handling(error_category=ErrorCategory.CONFIGURATION)
def setup_application(args:argparse.Namespace)->Tuple[Optional[MaggieAI],Dict[str,Any]]:
	config_manager=ConfigManager(args.config);config=config_manager.load();initialize_logging(config,args.debug);logger.info('Starting Maggie AI Assistant');logger.info(f"Running on Python {platform.python_version()}");logger.info(f"Process ID: {os.getpid()}");initialize_multiprocessing()
	try:maggie=MaggieAI(config_manager);register_signal_handlers(maggie);logger.info('Application setup completed successfully');return maggie,config
	except ImportError as e:record_error(message=f"Failed to import required module: {e}",exception=e,category=ErrorCategory.SYSTEM,severity=ErrorSeverity.CRITICAL,source='main.setup_application');return None,config
	except Exception as e:record_error(message=f"Error setting up application: {e}",exception=e,category=ErrorCategory.SYSTEM,severity=ErrorSeverity.CRITICAL,source='main.setup_application');return None,config
@with_error_handling(error_category=ErrorCategory.SYSTEM)
def setup_gui(maggie:MaggieAI)->Optional[Tuple[Any,Any]]:
	try:setup_pyside6_paths();from PySide6.QtWidgets import QApplication;from maggie.utils import get_main_window;MainWindow=get_main_window();app=QApplication(sys.argv);window=MainWindow(maggie);return window,app
	except ImportError as e:logger.error(f"Failed to import GUI modules: {e}");return None
	except Exception as e:logger.error(f"Error setting up GUI: {e}");return None
@with_error_handling(error_category=ErrorCategory.SYSTEM)
def start_maggie(args:argparse.Namespace,maggie:MaggieAI,config:Dict[str,Any])->int:
	logger.info('Starting Maggie AI core services');success=maggie.start()
	if not success:logger.error('Failed to start Maggie AI core services');return 1
	if not args.headless:
		try:
			gui_result=setup_gui(maggie)
			if gui_result is None:logger.error('GUI setup failed');maggie.shutdown();return 1
			window,app=gui_result
			if hasattr(maggie,'set_gui')and callable(getattr(maggie,'set_gui')):maggie.set_gui(window)
			window.show();return app.exec()
		except Exception as e:logger.error(f"Error starting GUI: {e}");maggie.shutdown();return 1
	else:
		logger.info('Running in headless mode')
		try:
			while maggie.state!=State.SHUTDOWN:time.sleep(1)
			return 0
		except KeyboardInterrupt:logger.info('Keyboard interrupt received, shutting down');maggie.shutdown();return 0
def main()->int:
	try:
		args=parse_arguments();maggie,config=setup_application(args)
		if maggie is None:logger.error('Failed to set up application');return 1
		return start_maggie(args,maggie,config)
	except KeyboardInterrupt:logger.info('\nApplication interrupted by user');return 1
	except Exception as e:logger.critical(f"Unexpected error in main: {e}",exc_info=True);return 1
if __name__=='__main__':sys.exit(main())