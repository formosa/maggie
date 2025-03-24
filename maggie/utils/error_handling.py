import sys,traceback,logging
from typing import Any,Callable,Optional,TypeVar,Dict,Union,List,Tuple,cast
T=TypeVar('T')
logger=logging.getLogger(__name__)
def safe_execute(func:Callable[...,T],*args:Any,default_return:Optional[T]=None,error_message:str='Error executing function',critical:bool=False,event_bus:Any=None,**kwargs:Any)->T:
	try:return func(*args,**kwargs)
	except Exception as e:
		exc_type,exc_value,exc_traceback=sys.exc_info();tb=traceback.extract_tb(exc_traceback)
		if tb:
			filename,line,func_name,text=tb[-1];error_detail=f"{error_message}: {e} in {filename}:{line} (function: {func_name})"
			if critical:logger.critical(error_detail)
			else:logger.error(error_detail)
			if event_bus:
				error_data={'message':str(e),'source':filename,'line':line,'function':func_name,'type':exc_type.__name__,'context':error_message}
				try:event_bus.publish('error_logged',error_data)
				except Exception as event_error:logger.error(f"Failed to publish error event: {event_error}")
		elif critical:logger.critical(f"{error_message}: {e}")
		else:logger.error(f"{error_message}: {e}")
		return default_return if default_return is not None else cast(T,None)
def retry_operation(max_attempts:int=3,retry_delay:float=1.,exponential_backoff:bool=True,allowed_exceptions:Tuple[Exception,...]=(Exception,))->Callable:
	def decorator(func:Callable)->Callable:
		def wrapper(*args:Any,**kwargs:Any)->Any:
			last_exception=None
			for attempt in range(1,max_attempts+1):
				try:return func(*args,**kwargs)
				except allowed_exceptions as e:
					last_exception=e
					if attempt==max_attempts:logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}");raise
					delay=retry_delay
					if exponential_backoff:delay=retry_delay*2**(attempt-1)
					logger.warning(f"Attempt {attempt}/{max_attempts} for {func.__name__} failed: {e}. Retrying in {delay:.2f}s");import time;time.sleep(delay)
			if last_exception:raise last_exception
			return None
		return wrapper
	return decorator
def get_event_bus()->Optional[Any]:
	try:from maggie.utils.service_locator import ServiceLocator;return ServiceLocator.get('event_bus')
	except ImportError:logger.warning("ServiceLocator not available, can't get event_bus");return None
	except Exception as e:logger.error(f"Error getting event_bus: {e}");return None