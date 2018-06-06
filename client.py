#partly implementation. not the best code
import socket
import json
import platform
import getpass

def find_open_ports():
	ports = []
	for port in range(5555, 5590):
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		res = sock.connect_ex(('localhost', port))
		if res == 0:
			ports.append(port)
			sock.close()
	return ports

PORT = 5555
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
s.connect(("localhost", PORT))
msg = s.recv(4096).decode("utf8")
msg = json.loads(msg)
print("Got message from server:\n %s" % msg)
#now get all the system info
msg = {}
msg["os_type"] = platform.platform()
msg["user"] = getpass.getuser()
msg["ports"] = str(find_open_ports())
msg = str(msg)
msg = json.dumps(msg)
print("Going to send %s" % msg)
s.send(bytes(msg,"utf8"))
msg = s.recv(4096).decode("utf8")
if msg == "disconnect":
	s.send(bytes("ok","utf8"))
else:
	print("Got different message from server than expected: %s" % msg)
print("Closing connection...")
s.close()








