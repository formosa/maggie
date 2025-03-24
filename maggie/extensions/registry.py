import os,sys,importlib,importlib.util,pkgutil
from typing import Dict,Type,Any,Optional,List,Tuple,cast
from maggie.extensions.base import ExtensionBase
from maggie.utils.error_handling import safe_execute,with_error_handling,ErrorCategory,ErrorSeverity,record_error
from maggie.utils.logging import ComponentLogger,log_operation,logging_context
class ExtensionRegistry:
	def __init__(self,extensions_path:Optional[str]=None):
		(self._registry):Dict[str,Type[ExtensionBase]]={};self.logger=ComponentLogger('ExtensionRegistry')
		if extensions_path is None:package_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));self._extensions_path=os.path.join(package_dir,'extensions')
		else:self._extensions_path=extensions_path
		os.makedirs(self._extensions_path,exist_ok=True);self.event_bus=self._get_event_bus()
	def _get_event_bus(self)->Optional[Any]:
		try:from maggie.service.locator import ServiceLocator;return ServiceLocator.get('event_bus')
		except(ImportError,Exception)as e:self.logger.warning(f"Could not access event bus: {e}");return None
	@log_operation(component='ExtensionRegistry')
	@with_error_handling(error_category=ErrorCategory.EXTENSION)
	def discover_extensions(self)->Dict[str,Type[ExtensionBase]]:
		self._registry={}
		if self._extensions_path not in sys.path:sys.path.append(self._extensions_path)
		with logging_context(component='ExtensionRegistry',operation='discover_extensions'):
			for(_,module_name,is_pkg)in pkgutil.iter_modules([self._extensions_path]):
				if is_pkg:self._process_extension_package(module_name)
			extension_count=len(self._registry);self.logger.info(f"Discovered {extension_count} extensions: {', '.join(self._registry.keys())}");return self._registry
	def _process_extension_package(self,module_name:str)->None:
		try:
			extension_module=importlib.import_module(module_name);main_module_name=f"{module_name}.{module_name}"
			try:extension_module=importlib.import_module(main_module_name);self._extract_extension_classes(extension_module,module_name)
			except ImportError:self.logger.warning(f"Could not import main module for extension: {module_name}")
		except ImportError:self.logger.warning(f"Failed to import extension package {module_name}");self._publish_extension_error(module_name,f"Failed to import extension package")
		except Exception as e:self.logger.error(f"Unexpected error processing extension package {module_name}");self._publish_extension_error(module_name,str(e))
	def _extract_extension_classes(self,module:Any,module_name:str)->None:
		extension_found=False
		for attr_name in dir(module):
			attr=getattr(module,attr_name)
			if isinstance(attr,type)and issubclass(attr,ExtensionBase)and attr is not ExtensionBase:self._registry[module_name]=attr;self.logger.info(f"Discovered extension: {module_name}");extension_found=True;break
		if not extension_found:self.logger.debug(f"No valid extension class found in module: {module_name}")
	def _publish_extension_error(self,extension_name:str,error_message:str)->None:
		if self.event_bus:self.event_bus.publish('error_logged',{'source':'ExtensionRegistry','extension':extension_name,'message':f"Extension error: {error_message}",'category':ErrorCategory.EXTENSION.value,'severity':ErrorSeverity.WARNING.value})
	@log_operation(component='ExtensionRegistry')
	def get_extension_class(self,extension_name:str)->Optional[Type[ExtensionBase]]:return self._registry.get(extension_name)
	@log_operation(component='ExtensionRegistry')
	@with_error_handling(error_category=ErrorCategory.EXTENSION)
	def instantiate_extension(self,extension_name:str,event_bus:Any,config:Dict[str,Any])->Optional[ExtensionBase]:
		extension_class=self.get_extension_class(extension_name)
		if extension_class is None:self.logger.error(f"Extension not found: {extension_name}");self._publish_extension_error(extension_name,'Extension not found');return None
		try:extension_instance=extension_class(event_bus,config);self.logger.info(f"Instantiated extension: {extension_name}");return extension_instance
		except Exception as e:self.logger.error(f"Error instantiating extension {extension_name}: {e}");self._publish_extension_error(extension_name,str(e));return None
	def get_available_extensions(self)->List[str]:return list(self._registry.keys())
	@log_operation(component='ExtensionRegistry')
	@with_error_handling(error_category=ErrorCategory.EXTENSION,error_severity=ErrorSeverity.WARNING)
	def reload_extension(self,extension_name:str)->bool:
		if extension_name not in self._registry:self.logger.warning(f"Cannot reload unknown extension: {extension_name}");return False
		try:
			module_name=f"{extension_name}.{extension_name}"
			if module_name in sys.modules:module=sys.modules[module_name];importlib.reload(module);self._extract_extension_classes(module,extension_name);self.logger.info(f"Reloaded extension: {extension_name}");return True
			else:self.logger.warning(f"Extension module {module_name} not loaded, cannot reload");return False
		except Exception as e:self.logger.error(f"Error reloading extension {extension_name}: {e}");self._publish_extension_error(extension_name,f"Failed to reload: {e}");return False