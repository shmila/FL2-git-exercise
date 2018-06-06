#redis client
import socket

PORT = 5555
SIZE_MSG = 4096
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
s.connect(("localhost", PORT))
while True:
	msg = s.recv(SIZE_MSG).decode("utf8")
	if msg == 'bye bye':
		#close connection
		print("Closing connection...")
		s.close()
		break
	res = input(msg)
	s.send(bytes(res,'utf8'))

