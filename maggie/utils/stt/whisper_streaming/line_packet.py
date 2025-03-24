#!/usr/bin/env python3
PACKET_SIZE=65536
def send_one_line(socket,text,pad_zeros=False):
	text=text.replace('\x00','\n');lines=text.splitlines();first_line=''if len(lines)==0 else lines[0];data=first_line.encode('utf-8',errors='replace')+b'\n'+(b'\x00'if pad_zeros else b'')
	for offset in range(0,len(data),PACKET_SIZE):
		bytes_remaining=len(data)-offset
		if bytes_remaining<PACKET_SIZE:padding_length=PACKET_SIZE-bytes_remaining;packet=data[offset:]+(b'\x00'*padding_length if pad_zeros else b'')
		else:packet=data[offset:offset+PACKET_SIZE]
		socket.sendall(packet)
def receive_one_line(socket):
	data=b''
	while True:
		packet=socket.recv(PACKET_SIZE)
		if not packet:return None
		data+=packet
		if b'\x00'in packet:break
	text=data.decode('utf-8',errors='replace').strip('\x00');lines=text.split('\n');return lines[0]+'\n'
def receive_lines(socket):
	try:data=socket.recv(PACKET_SIZE)
	except BlockingIOError:return[]
	if not data:return None
	text=data.decode('utf-8',errors='replace').strip('\x00');lines=text.split('\n')
	if len(lines)==1 and not lines[0]:return None
	return lines