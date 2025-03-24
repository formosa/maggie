import queue,threading,time
from typing import Dict,Any,Optional,List,Callable,Set,Tuple,Union
from loguru import logger
from maggie.utils.error_handling import safe_execute,ErrorCategory
from maggie.utils.logging import log_operation,ComponentLogger
class EventPriority:HIGH=0;NORMAL=10;LOW=20;BACKGROUND=30
class EventBus:
	def __init__(self):(self.subscribers):Dict[str,List[Tuple[int,Callable]]]={};self.queue=queue.PriorityQueue();self.running=False;self._worker_thread=None;self._lock=threading.RLock();self.logger=ComponentLogger('EventBus')
	@log_operation(component='EventBus')
	def subscribe(self,event_type:str,callback:Callable,priority:int=EventPriority.NORMAL)->None:
		with self._lock:
			if event_type not in self.subscribers:self.subscribers[event_type]=[]
			self.subscribers[event_type].append((priority,callback));self.subscribers[event_type].sort(key=lambda x:x[0]);self.logger.debug(f"Subscription added for event type: {event_type}")
	@log_operation(component='EventBus')
	def unsubscribe(self,event_type:str,callback:Callable)->bool:
		with self._lock:
			if event_type not in self.subscribers:return False
			for(i,(_,cb))in enumerate(self.subscribers[event_type]):
				if cb==callback:self.subscribers[event_type].pop(i);self.logger.debug(f"Unsubscribed from event type: {event_type}");return True
			return False
	@log_operation(component='EventBus',log_args=True)
	def publish(self,event_type:str,data:Any=None,priority:int=EventPriority.NORMAL)->None:self.queue.put((priority,(event_type,data)))
	@log_operation(component='EventBus')
	def start(self)->bool:
		with self._lock:
			if self.running:return False
			self.running=True;self._worker_thread=threading.Thread(target=self._process_events,name='EventBusThread',daemon=True);self._worker_thread.start();self.logger.info('Event bus started');return True
	@log_operation(component='EventBus')
	def stop(self)->bool:
		with self._lock:
			if not self.running:return False
			self.running=False;self.queue.put((0,None))
			if self._worker_thread:self._worker_thread.join(timeout=2.)
			self.logger.info('Event bus stopped');return True
	def _process_events(self)->None:
		while self.running:
			try:
				priority,event=self.queue.get(timeout=.05)
				if event is None:break
				event_type,data=event;self._dispatch_event(event_type,data);self.queue.task_done()
			except queue.Empty:time.sleep(.001)
			except Exception as e:self.logger.error(f"Error processing events: {e}")
	def _dispatch_event(self,event_type:str,data:Any)->None:
		with self._lock:
			if event_type in self.subscribers:
				for(_,callback)in self.subscribers[event_type]:
					try:callback(data)
					except Exception as e:error_msg=f"Error in event handler for {event_type}: {e}";self.logger.error(error_msg);self.publish('error_logged',{'message':error_msg,'event_type':event_type,'source':'event_bus'},priority=EventPriority.HIGH)
class EventEmitter:
	def __init__(self,event_bus:EventBus):self.event_bus=event_bus;self.logger=ComponentLogger(self.__class__.__name__)
	def emit(self,event_type:str,data:Any=None,priority:int=EventPriority.NORMAL)->None:self.event_bus.publish(event_type,data,priority)
class EventListener:
	def __init__(self,event_bus:EventBus):self.event_bus=event_bus;self.logger=ComponentLogger(self.__class__.__name__);(self.subscriptions):Set[Tuple[str,Callable]]=set()
	def listen(self,event_type:str,callback:Callable,priority:int=EventPriority.NORMAL)->None:self.event_bus.subscribe(event_type,callback,priority);self.subscriptions.add((event_type,callback))
	def stop_listening(self,event_type:str=None,callback:Callable=None)->None:
		if event_type is None and callback is None:
			for(event_type,callback)in list(self.subscriptions):self.event_bus.unsubscribe(event_type,callback);self.subscriptions.remove((event_type,callback))
		elif callback is None:
			for(evt_type,cb)in list(self.subscriptions):
				if evt_type==event_type:self.event_bus.unsubscribe(evt_type,cb);self.subscriptions.remove((evt_type,cb))
		elif event_type is None:
			for(evt_type,cb)in list(self.subscriptions):
				if cb==callback:self.event_bus.unsubscribe(evt_type,cb);self.subscriptions.remove((evt_type,cb))
		else:self.event_bus.unsubscribe(event_type,callback);self.subscriptions.remove((event_type,callback))