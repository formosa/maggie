import os,json,uuid,platform,logging,threading,traceback,sys
from datetime import datetime
from typing import Dict,Any,Optional
import psutil,requests,torch
class MaggieTelemetryManager:
	def __init__(self,config_path:Optional[str]=None):self._instance_id=str(uuid.uuid4());self._config=self._load_telemetry_config(config_path);self._logger=self._configure_logging()
	def _load_telemetry_config(self,config_path:Optional[str]=None)->Dict[str,Any]:
		default_config={'logging':{'level':'INFO','filepath':'logs/maggie_telemetry.log'},'telemetry':{'opt_in':False,'endpoint':'https://telemetry.maggieai.com/report'}}
		if config_path and os.path.exists(config_path):
			try:
				with open(config_path,'r')as f:user_config=json.load(f);return{**default_config,**user_config}
			except(IOError,json.JSONDecodeError):self._logger.warning('Invalid telemetry config. Using defaults.')
		return default_config
	def _configure_logging(self)->logging.Logger:log_config=self._config.get('logging',{});log_level=getattr(logging,log_config.get('level','INFO').upper());log_filepath=log_config.get('filepath','logs/maggie_telemetry.log');os.makedirs(os.path.dirname(log_filepath),exist_ok=True);logging.basicConfig(level=log_level,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',handlers=[logging.FileHandler(log_filepath),logging.StreamHandler()]);return logging.getLogger('MaggieTelemetry')
	def capture_system_snapshot(self)->Dict[str,Any]:
		system_info={'instance_id':self._instance_id,'timestamp':datetime.utcnow().isoformat(),'system':{'os':platform.system(),'release':platform.release(),'architecture':platform.machine(),'python_version':platform.python_version()},'hardware':{'cpu':{'name':platform.processor(),'physical_cores':os.cpu_count()},'memory':{'total_gb':round(psutil.virtual_memory().total/1024**3,2)}}}
		try:system_info['hardware']['gpu']={'cuda_available':torch.cuda.is_available(),'device_name':torch.cuda.get_device_name(0)if torch.cuda.is_available()else None,'cuda_version':torch.version.cuda if torch.cuda.is_available()else None}
		except ImportError:system_info['hardware']['gpu']={'cuda_available':False}
		return system_info
	def log_installation_event(self,event_type:str,details:Dict[str,Any]):event_log={'event_type':event_type,'timestamp':datetime.utcnow().isoformat(),**details};self._logger.info(f"Event Logged: {json.dumps(event_log)}")
	def submit_anonymous_telemetry(self,event_data:Dict[str,Any])->bool:
		if not self._config['telemetry']['opt_in']:return False
		try:response=requests.post(self._config['telemetry']['endpoint'],json=event_data,timeout=5);return response.status_code==200
		except Exception as e:self._logger.error(f"Telemetry submission failed: {e}");return False
def global_exception_handler(exc_type,exc_value,exc_traceback):error_details={'type':str(exc_type.__name__),'message':str(exc_value),'traceback':''.join(traceback.format_tb(exc_traceback))};telemetry_manager=MaggieTelemetryManager();telemetry_manager._logger.critical(f"Unhandled Exception: {json.dumps(error_details)}");telemetry_manager.submit_anonymous_telemetry({'event_type':'unhandled_exception',**error_details})
sys.excepthook=global_exception_handler