import sys,time
from enum import Enum
from typing import Dict,Any,Optional,List,Callable
from main import add_pyside6_paths
add_pyside6_paths()
from PySide6.QtWidgets import QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QTextEdit,QPushButton,QLabel,QSplitter,QTabWidget,QSizePolicy,QListWidget,QListWidgetItem,QGroupBox,QFrame,QStatusBar,QApplication,QLineEdit
from PySide6.QtCore import Qt,Signal,QTimer,QMetaObject,Q_ARG,QThread
from PySide6.QtGui import QFont,QColor,QIcon,QKeySequence,QShortcut,QFocusEvent
from loguru import logger
__all__=['MainWindow']
class State(Enum):IDLE='idle';STARTING='starting';RUNNING='running';PAUSED='paused';STOPPED='stopped';ERROR='error'
class QVariant:
	def __init__(self,value=None):self.value=value
	def value(self):return self.value
class InputField(QLineEdit):
	state_change_requested=Signal(str)
	def __init__(self,parent=None,submit_callback=None):super().__init__(parent);self.stt_mode=True;self.submit_callback=submit_callback;self.setPlaceholderText('Speak or type your message here...');self.intermediate_text='';self.returnPressed.connect(self.on_return_pressed)
	def focusInEvent(self,event:QFocusEvent)->None:
		super().focusInEvent(event);self.stt_mode=False;self.setStyleSheet('background-color: white; color: black;');self.state_change_requested.emit('ACTIVE');main_window=self.window()
		if hasattr(main_window,'maggie_ai')and main_window.maggie_ai:main_window.maggie_ai.event_bus.publish('pause_transcription')
	def focusOutEvent(self,event:QFocusEvent)->None:
		super().focusOutEvent(event);self.stt_mode=True;self.update_appearance_for_state('IDLE')
		if not self.text().strip():
			main_window=self.window()
			if hasattr(main_window,'maggie_ai')and main_window.maggie_ai:main_window.maggie_ai.event_bus.publish('resume_transcription')
	def on_return_pressed(self)->None:
		if self.submit_callback and self.text().strip():self.submit_callback(self.text());self.clear();self.intermediate_text=''
	def update_appearance_for_state(self,state:str)->None:
		if state=='IDLE'and self.stt_mode:self.setStyleSheet('background-color: lightgray;');self.setReadOnly(True)
		else:
			self.setReadOnly(False)
			if not self.hasFocus():self.setStyleSheet('background-color: white;')
	def update_intermediate_text(self,text:str)->None:
		if self.stt_mode and not self.hasFocus():self.intermediate_text=text;self.setText(text);self.setStyleSheet('background-color: white; color: gray;')
	def set_final_text(self,text:str)->None:
		if self.stt_mode and not self.hasFocus():
			self.setText(text);self.setStyleSheet('background-color: white; color: black;');auto_submit=self.window().maggie_ai.config.get('stt',{}).get('auto_submit',False)
			if auto_submit and self.submit_callback and text.strip():self.submit_callback(text);self.clear();self.intermediate_text=''
class MainWindow(QMainWindow):
	def __init__(self,maggie_ai):super().__init__();self.maggie_ai=maggie_ai;self.setWindowTitle('Maggie AI Assistant');self.setMinimumSize(900,700);self.is_shutting_down=False;self.central_widget=QWidget();self.setCentralWidget(self.central_widget);self.main_layout=QVBoxLayout(self.central_widget);self.status_bar=QStatusBar();self.setStatusBar(self.status_bar);self.status_label=QLabel('Status: IDLE');self.status_label.setStyleSheet('font-weight: bold;');self.status_bar.addPermanentWidget(self.status_label);self._create_main_layout();self.update_state('IDLE');self.log_event('Maggie AI Assistant UI initialized...');self.maggie_ai.event_bus.subscribe('state_changed',self._on_state_changed);self.maggie_ai.event_bus.subscribe('extension_completed',self._on_extension_completed);self.maggie_ai.event_bus.subscribe('extension_error',self._on_extension_error);self.maggie_ai.event_bus.subscribe('error_logged',self._on_error_logged);self._connect_stt_events();self.setup_shortcuts()
	def _connect_stt_events(self)->None:self.maggie_ai.event_bus.subscribe('intermediate_transcription',self._on_intermediate_transcription,priority=0);self.maggie_ai.event_bus.subscribe('final_transcription',self._on_final_transcription,priority=0);self.maggie_ai.event_bus.subscribe('pause_transcription',self._on_pause_transcription);self.maggie_ai.event_bus.subscribe('resume_transcription',self._on_resume_transcription)
	def _on_intermediate_transcription(self,text:str)->None:self.safe_update_gui(self.input_field.update_intermediate_text,text)
	def _on_final_transcription(self,text:str)->None:self.safe_update_gui(self.input_field.set_final_text,text)
	def _on_pause_transcription(self,_=None)->None:
		from maggie.utils.service_locator import ServiceLocator;stt_processor=ServiceLocator.get('stt_processor')
		if stt_processor:stt_processor.pause_streaming()
	def _on_resume_transcription(self,_=None)->None:
		from maggie.utils.service_locator import ServiceLocator;stt_processor=ServiceLocator.get('stt_processor')
		if stt_processor:stt_processor.resume_streaming()
	def _create_main_layout(self)->None:self.content_splitter=QSplitter(Qt.Orientation.Horizontal);self.main_layout.addWidget(self.content_splitter);self.left_panel=QWidget();self.left_layout=QVBoxLayout(self.left_panel);self.content_splitter.addWidget(self.left_panel);self.right_panel=QWidget();self.right_layout=QVBoxLayout(self.right_panel);self.content_splitter.addWidget(self.right_panel);self.content_splitter.setSizes([700,200]);self._create_log_sections();self._create_right_panel();self._create_control_panel()
	def _create_log_sections(self)->None:self.logs_splitter=QSplitter(Qt.Orientation.Vertical);self.left_layout.addWidget(self.logs_splitter);self.chat_section=QWidget();self.chat_layout=QVBoxLayout(self.chat_section);self.chat_layout.setContentsMargins(0,0,0,0);self.chat_group=QGroupBox('Chat');self.chat_group_layout=QVBoxLayout(self.chat_group);self.chat_log=QTextEdit();self.chat_log.setReadOnly(True);self.chat_group_layout.addWidget(self.chat_log);self.chat_layout.addWidget(self.chat_group);self.input_field=InputField(submit_callback=self._on_input_submitted);self.input_field.setFixedHeight(30);self.input_field.update_appearance_for_state('IDLE');self.input_field.state_change_requested.connect(self._on_input_state_change);self.chat_layout.addWidget(self.input_field);self.event_group=QGroupBox('Event Log');self.event_group_layout=QVBoxLayout(self.event_group);self.event_log=QTextEdit();self.event_log.setReadOnly(True);self.event_group_layout.addWidget(self.event_log);self.error_group=QGroupBox('Error Log');self.error_group_layout=QVBoxLayout(self.error_group);self.error_log=QTextEdit();self.error_log.setReadOnly(True);self.error_group_layout.addWidget(self.error_log);self.logs_splitter.addWidget(self.chat_section);self.logs_splitter.addWidget(self.event_group);self.logs_splitter.addWidget(self.error_group);self.logs_splitter.setSizes([400,150,150])
	def _create_right_panel(self)->None:self.state_group=QGroupBox('Current State');self.state_layout=QVBoxLayout(self.state_group);self.state_display=QLabel('IDLE');self.state_display.setAlignment(Qt.AlignmentFlag.AlignCenter);self.state_display.setStyleSheet('font-size: 18px; font-weight: bold;');self.state_layout.addWidget(self.state_display);self.right_layout.addWidget(self.state_group);self.extensions_group=QGroupBox('Extensions');self.extensions_layout=QVBoxLayout(self.extensions_group);self._create_extension_buttons();self.right_layout.addWidget(self.extensions_group);self.right_layout.addStretch()
	def _create_control_panel(self)->None:self.control_panel=QWidget();self.control_layout=QHBoxLayout(self.control_panel);self.main_layout.addWidget(self.control_panel);self.shutdown_button=QPushButton('Shutdown');self.shutdown_button.clicked.connect(self.on_shutdown_clicked);self.control_layout.addWidget(self.shutdown_button);self.sleep_button=QPushButton('Sleep');self.sleep_button.clicked.connect(self.on_sleep_clicked);self.control_layout.addWidget(self.sleep_button)
	def _on_input_state_change(self,requested_state:str)->None:
		if self.maggie_ai.state.name=='IDLE'and requested_state=='ACTIVE':self.maggie_ai._transition_to(self.maggie_ai.state.READY,'input_field_activated');self.log_event('State transition requested by input field')
	def _on_error_logged(self,error_data):
		if isinstance(error_data,dict):message=error_data.get('message','Unknown error');source=error_data.get('source','system');self.log_error(f"[{source}] {message}")
		else:self.log_error(str(error_data))
	def setup_shortcuts(self)->None:
		try:shortcut_config={'sleep':'Alt+S','shutdown':'Alt+Q','focus_input':'Alt+I'};sleep_shortcut=QShortcut(QKeySequence(shortcut_config['sleep']),self);sleep_shortcut.activated.connect(self.on_sleep_clicked);shutdown_shortcut=QShortcut(QKeySequence(shortcut_config['shutdown']),self);shutdown_shortcut.activated.connect(self.on_shutdown_clicked);input_shortcut=QShortcut(QKeySequence(shortcut_config['focus_input']),self);input_shortcut.activated.connect(lambda:self.input_field.setFocus());logger.debug('Keyboard shortcuts configured')
		except Exception as e:logger.error(f"Error setting up shortcuts: {e}")
	def _create_extension_buttons(self)->None:
		try:
			self._cleanup_extension_buttons();self.extension_buttons={}
			for extension_name in self.maggie_ai.extensions:
				display_name=extension_name.replace('_',' ').title();extension_button=QPushButton(display_name);extension_button.clicked.connect(lambda checked=False,name=extension_name:self.on_extension_clicked(name));self.extensions_layout.addWidget(extension_button);self.extension_buttons[extension_name]=extension_button
				if extension_name=='recipe_creator':
					try:recipe_shortcut=QShortcut(QKeySequence('Alt+R'),self);recipe_shortcut.activated.connect(lambda:self.on_extension_clicked('recipe_creator'))
					except Exception as e:logger.error(f"Error setting up recipe shortcut: {e}")
		except Exception as e:logger.error(f"Error creating extension buttons: {e}")
	def _on_state_changed(self,transition)->None:
		try:
			if not transition or not hasattr(transition,'to_state')or not hasattr(transition,'from_state'):logger.error('Invalid state transition object received');return
			to_state=getattr(transition,'to_state',None);from_state=getattr(transition,'from_state',None)
			if to_state is None or from_state is None:logger.error('Invalid state transition object: to_state or from_state is None');return
			to_state_name=getattr(to_state,'name','UNKNOWN');from_state_name=getattr(from_state,'name','UNKNOWN');trigger=getattr(transition,'trigger','UNKNOWN');self.update_state(to_state_name);self.input_field.update_appearance_for_state(to_state_name);self.log_event(f"State changed: {from_state_name} -> {to_state_name} (trigger: {trigger})")
		except Exception as e:logger.error(f"Error handling state transition: {e}")
	def _cleanup_extension_buttons(self)->None:
		try:
			if hasattr(self,'extension_buttons'):
				for button in self.extension_buttons.values():self.extensions_layout.removeWidget(button);button.deleteLater()
				self.extension_buttons.clear()
		except Exception as e:logger.error(f"Error cleaning up extension buttons: {e}")
	def update_state(self,state:State)->None:
		valid_states=['IDLE','STARTUP','READY','ACTIVE','BUSY','CLEANUP','SHUTDOWN']
		if state not in valid_states:logger.warning(f"Invalid state: {state}. Defaulting to IDLE.");state='IDLE'
		self.state_display.setText(state);self.status_label.setText(f"Status: {state}");color_map={'IDLE':'lightgray','STARTUP':'lightblue','READY':'lightgreen','ACTIVE':'yellow','BUSY':'orange','CLEANUP':'pink','SHUTDOWN':'red'};color=color_map.get(state,'white');self.state_display.setStyleSheet(f"font-size: 18px; font-weight: bold; background-color: {color}; padding: 5px;");self.input_field.update_appearance_for_state(state);logger.debug(f"GUI state updated to: {state}")
	def refresh_extensions(self)->None:self._create_extension_buttons();self.log_event('Extension list updated')
	def log_chat(self,message:str,is_user:bool=False)->None:timestamp=time.strftime('%H:%M:%S');prefix='user'if is_user else'Maggie';color='blue'if is_user else'green';self.chat_log.append(f'<span style="color:gray">[{timestamp}]</span> <span style="color:{color}"><b>&lt; {prefix} &gt;</b></span> {message}')
	def log_event(self,event:str)->None:timestamp=time.strftime('%H:%M:%S');self.event_log.append(f'<span style="color:gray">[{timestamp}]</span> {event}');logger.debug(f"Event logged: {event}")
	def show_download_progress(self,progress_data):
		item=progress_data.get('item','file');percent=progress_data.get('percent',0);status=f"Downloading {item}: {percent}% complete";self.status_label.setText(status)
		if percent>=100:QTimer.singleShot(3000,lambda:self.status_label.setText(f"Status: {self.state_display.text()}"))
	def log_error(self,error:str)->None:
		timestamp=time.strftime('%H:%M:%S');formatted_error=f'<span style="color:gray">[{timestamp}]</span> <span style="color:red"><b>ERROR:</b></span> {error}';self.error_log.append(formatted_error);current_sizes=self.logs_splitter.sizes()
		if current_sizes[2]<100:self.logs_splitter.setSizes([current_sizes[0],current_sizes[1],200])
		logger.error(f"Error logged in GUI: {error}")
	def on_shutdown_clicked(self)->None:self.log_event('Shutdown requested');self.maggie_ai.shutdown();logger.info('Shutdown initiated from GUI')
	def on_sleep_clicked(self)->None:self.log_event('Sleep requested');self.maggie_ai.timeout();logger.info('Sleep initiated from GUI')
	def on_extension_clicked(self,extension_name:str)->None:
		self.log_event(f"Extension requested: {extension_name}")
		if extension_name in self.maggie_ai.extensions:extension=self.maggie_ai.extensions[extension_name];self.maggie_ai.process_command(extension=extension);logger.info(f"Extension '{extension_name}' activated from GUI")
	def _on_input_submitted(self,text:str)->None:
		if not text.strip():return
		self.log_chat(text,is_user=True)
		if self.maggie_ai.state.name=='IDLE':self.maggie_ai._transition_to(self.maggie_ai.state.READY,'user_input')
		self.maggie_ai.event_bus.publish('command_detected',text);logger.debug(f"User input submitted: {text}")
	def closeEvent(self,event)->None:self.log_event('Window close requested, shutting down');self.is_shutting_down=True;self.maggie_ai.shutdown();QTimer.singleShot(2000,lambda:event.accept());logger.info('GUI window closed, shutdown initiated')
	def safe_update_gui(self,func:Callable,*args,**kwargs)->None:
		if QThread.currentThread()==self.thread():
			try:func(*args,**kwargs);return
			except Exception as e:logger.error(f"Error calling GUI method directly: {e}");return
		try:
			q_args=[]
			for arg in args:
				try:
					if isinstance(arg,(int,float,bool,str)):q_args.append(Q_ARG(type(arg),arg))
					else:q_args.append(Q_ARG(QVariant,QVariant(arg)))
				except Exception as e:logger.debug(f"Error converting argument to Q_ARG: {e}");q_args.append(Q_ARG(str,str(arg)))
			QMetaObject.invokeMethod(self,func.__name__,Qt.ConnectionType.QueuedConnection,*q_args)
		except Exception as e:logger.error(f"Error invoking GUI method {func.__name__}: {e}")
	def _on_extension_completed(self,extension_name:str)->None:self.log_event(f"Extension completed: {extension_name}");self.update_state('READY')
	def _on_extension_error(self,extension_name:str)->None:self.log_error(f"Error in extension: {extension_name}");self.update_state('READY')
	def update_stt_text(self,text:str)->None:
		if self.input_field.stt_mode:self.input_field.setText(text)