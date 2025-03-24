import torch,numpy as np
class VADIterator:
	def __init__(self,model,threshold:float=.5,sampling_rate:int=16000,min_silence_duration_ms:int=500,speech_pad_ms:int=100):
		self.model=model;self.threshold=threshold;self.sampling_rate=sampling_rate
		if sampling_rate not in[8000,16000]:raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')
		self.min_silence_samples=sampling_rate*min_silence_duration_ms/1000;self.speech_pad_samples=sampling_rate*speech_pad_ms/1000;self.reset_states()
	def reset_states(self):self.model.reset_states();self.triggered=False;self.temp_end=0;self.current_sample=0
	def __call__(self,x,return_seconds=False):
		if not torch.is_tensor(x):
			try:x=torch.Tensor(x)
			except:raise TypeError('Audio cannot be casted to tensor. Cast it manually')
		window_size_samples=len(x[0])if x.dim()==2 else len(x);self.current_sample+=window_size_samples;speech_prob=self.model(x,self.sampling_rate).item()
		if speech_prob>=self.threshold and self.temp_end:self.temp_end=0
		if speech_prob>=self.threshold and not self.triggered:self.triggered=True;speech_start=self.current_sample-self.speech_pad_samples;return{'start':int(speech_start)if not return_seconds else round(speech_start/self.sampling_rate,1)}
		if speech_prob<self.threshold-.15 and self.triggered:
			if not self.temp_end:self.temp_end=self.current_sample
			if self.current_sample-self.temp_end<self.min_silence_samples:return None
			else:speech_end=self.temp_end+self.speech_pad_samples;self.temp_end=0;self.triggered=False;return{'end':int(speech_end)if not return_seconds else round(speech_end/self.sampling_rate,1)}
		return None
class FixedVADIterator(VADIterator):
	def reset_states(self):super().reset_states();self.buffer=np.array([],dtype=np.float32)
	def __call__(self,x,return_seconds=False):
		self.buffer=np.append(self.buffer,x);ret=None
		while len(self.buffer)>=512:
			r=super().__call__(self.buffer[:512],return_seconds=return_seconds);self.buffer=self.buffer[512:]
			if ret is None:ret=r
			elif r is not None:
				if'end'in r:ret['end']=r['end']
				if'start'in r and'end'in ret:del ret['end']
		return ret if ret!={}else None
if __name__=='__main__':import torch;model,_=torch.hub.load(repo_or_dir='snakers4/silero-vad',model='silero_vad');vac=FixedVADIterator(model);audio_buffer=np.array([0]*512,dtype=np.float32);vac(audio_buffer);audio_buffer=np.array([0]*(512-1),dtype=np.float32);vac(audio_buffer)