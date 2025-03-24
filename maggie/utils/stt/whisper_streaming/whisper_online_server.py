#!/usr/bin/env python3
import sys,argparse,os,logging,numpy as np,socket,io,soundfile
from whisper_streaming.whisper_online import load_audio_chunk,asr_factory,set_logging,add_shared_args
from whisper_streaming.line_packet import send_one_line,receive_one_line,receive_lines,PACKET_SIZE
logger=logging.getLogger(__name__)
class Connection:
	PACKET_SIZE=32000*5*60
	def __init__(self,conn):self.conn=conn;self.last_line='';self.conn.setblocking(True)
	def send(self,line):
		if line==self.last_line:return
		send_one_line(self.conn,line);self.last_line=line
	def receive_lines(self):in_line=receive_lines(self.conn);return in_line
	def non_blocking_receive_audio(self):
		try:r=self.conn.recv(self.PACKET_SIZE);return r
		except ConnectionResetError:return None
class ServerProcessor:
	def __init__(self,c,online_asr_proc,min_chunk):self.connection=c;self.online_asr_proc=online_asr_proc;self.min_chunk=min_chunk;self.last_end=None;self.is_first=True
	def receive_audio_chunk(self):
		out=[];minlimit=self.min_chunk*16000
		while sum(len(x)for x in out)<minlimit:
			raw_bytes=self.connection.non_blocking_receive_audio()
			if not raw_bytes:break
			sf_file=soundfile.SoundFile(io.BytesIO(raw_bytes),channels=1,endian='LITTLE',samplerate=16000,subtype='PCM_16',format='RAW');audio,_=librosa.load(sf_file,sr=16000,dtype=np.float32);out.append(audio)
		if not out:return None
		conc=np.concatenate(out)
		if self.is_first and len(conc)<minlimit:return None
		self.is_first=False;return np.concatenate(out)
	def format_output_transcript(self,o):
		if o[0]is not None:
			beg,end=o[0]*1000,o[1]*1000
			if self.last_end is not None:beg=max(beg,self.last_end)
			self.last_end=end;print('%1.0f %1.0f %s'%(beg,end,o[2]),flush=True,file=sys.stderr);return'%1.0f %1.0f %s'%(beg,end,o[2])
		else:logger.debug('No text in this segment');return None
	def send_result(self,o):
		msg=self.format_output_transcript(o)
		if msg is not None:self.connection.send(msg)
	def process(self):
		self.online_asr_proc.init()
		while True:
			a=self.receive_audio_chunk()
			if a is None:break
			self.online_asr_proc.insert_audio_chunk(a);o=self.online_asr_proc.process_iter()
			try:self.send_result(o)
			except BrokenPipeError:logger.info('Broken pipe -- connection closed?');break
def main():
	parser=argparse.ArgumentParser();parser.add_argument('--host',type=str,default='localhost',help='Host to bind the server to');parser.add_argument('--port',type=int,default=43007,help='Port to listen on');parser.add_argument('--warmup-file',type=str,dest='warmup_file',help='Path to a speech audio wav file to warm up Whisper');add_shared_args(parser);args=parser.parse_args();set_logging(args,logger,other='');size=args.model;language=args.lan;asr,online=asr_factory(args);min_chunk=args.min_chunk_size;msg='Whisper is not warmed up. The first chunk processing may take longer.'
	if args.warmup_file:
		if os.path.isfile(args.warmup_file):from whisper_streaming.whisper_online import load_audio_chunk;a=load_audio_chunk(args.warmup_file,0,1);asr.transcribe(a);logger.info('Whisper is warmed up.')
		else:logger.critical('The warm up file is not available. '+msg);sys.exit(1)
	else:logger.warning(msg)
	with socket.socket(socket.AF_INET,socket.SOCK_STREAM)as s:
		s.bind((args.host,args.port));s.listen(1);logger.info(f"Listening on {args.host}:{args.port}")
		while True:conn,addr=s.accept();logger.info(f"Connected to client on {addr}");connection=Connection(conn);proc=ServerProcessor(connection,online,args.min_chunk_size);proc.process();conn.close();logger.info('Connection to client closed')
	logger.info('Connection closed, terminating.')
if __name__=='__main__':import librosa;main()