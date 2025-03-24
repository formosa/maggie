import os,argparse,sys,platform,multiprocessing,time,yaml
from typing import Dict,Any,Optional,List,Tuple
from maggie.utils.logging import LoggingManager,logger
from maggie.utils.error_handling import safe_execute,ErrorCategory,ErrorSeverity,ErrorRegistry,ErrorContext,record_error
def parse_arguments()->argparse.Namespace:parser=argparse.ArgumentParser(description='Maggie AI Assistant',formatter_class=argparse.ArgumentDefaultsHelpFormatter);parser.add_argument('--config',type=str,default='config.yaml',help='Path to configuration file');parser.add_argument('--debug',action='store_true',help='Enable debug logging');parser.add_argument('--verify',action='store_true',help='Verify system configuration without starting the assistant');parser.add_argument('--create-template',action='store_true',help="Create the recipe template file if it doesn't exist");parser.add_argument('--optimize',action='store_true',help='Optimize configuration for detected hardware');parser.add_argument('--headless',action='store_true',help='Run in headless mode without GUI');return parser.parse_args()
def load_configuration(config_path:str)->Dict[str,Any]:
	if not os.path.exists(config_path):logger.warning(f"Config file not found: {config_path}, using defaults");return{}
	try:
		with open(config_path,'r')as f:
			config=yaml.safe_load(f)
			if config is None:logger.warning(f"Empty config file: {config_path}, using defaults");return{}
			return config
	except Exception as e:error_registry=ErrorRegistry.get_instance();error_ctx=error_registry.create_error(code='CONFIG_LOAD_ERROR',details={'path':config_path,'error':str(e)},exception=e,source='main.load_configuration');error_ctx.log(publish=True);return{}
def setup_application(args:argparse.Namespace)->Tuple[Any,Dict[str,Any]]:
	os.makedirs('logs',exist_ok=True)
	try:multiprocessing.set_start_method('spawn');logger.info("Set multiprocessing start method to 'spawn'")
	except RuntimeError:pass
	config=load_configuration(args.config);logging_manager=LoggingManager.initialize(config);logging_manager.setup_global_exception_handler();logger.info('Starting Maggie AI Assistant');logger.info(f"Running on Python {platform.python_version()}");logger.info(f"Process ID: {os.getpid()}")
	if args.optimize:
		if optimize_system():logger.info('System optimized for performance')
		else:logger.warning('System optimization failed')
	if args.verify or args.create_template:return None,config
	if not verify_system():logger.warning('System verification failed, but attempting to start anyway')
	try:
		if not args.headless:add_pyside6_paths()
		from maggie.core import MaggieAI,State;maggie=MaggieAI(config);register_signal_handlers(maggie);event_bus=maggie.event_bus;logging_manager.add_event_bus_handler(event_bus);return maggie,config
	except ImportError as e:record_error(message=f"Failed to import required module: {e}",exception=e,category=ErrorCategory.SYSTEM,severity=ErrorSeverity.CRITICAL,source='main.setup_application');return None,config
def main()->int:
	args=parse_arguments()
	try:
		maggie,config=setup_application(args)
		if args.create_template:
			create_recipe_template()
			if not args.verify:return 0
		if args.verify:
			if verify_system():logger.info('System verification successful');return 0
			else:logger.error('System verification failed');return 1
		if maggie is None:logger.error('Failed to set up application');return 1
		return start_maggie(args,maggie,config)
	except KeyboardInterrupt:logger.info('\nApplication interrupted by user');return 1
	except Exception as e:logger.critical(f"Unexpected error in main: {e}",exc_info=True);return 1
def start_maggie(args:argparse.Namespace,maggie:Any,config:Dict[str,Any])->int:
	success=maggie.start()
	if not success:logger.error('Failed to start Maggie AI core services');return 1
	if not args.headless:
		try:from PySide6.QtWidgets import QApplication;from maggie.utils.gui import MainWindow;app=QApplication(sys.argv);window=MainWindow(maggie);maggie.set_gui(window);window.show();return app.exec()
		except Exception as e:logger.error(f"Error starting GUI: {e}");maggie.shutdown();return 1
	else:
		from maggie.core import State;logger.info('Running in headless mode')
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