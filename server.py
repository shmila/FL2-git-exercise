#partly implementation. not the best code
import socket
from threading import Thread
import json


def handle_client(client,client_address):
	address = client_address[0]
	info_types = "{'command':  ['os_type', 'user', 'ports']}"
	msg = json.dumps(info_types)
	print("sending %s" % msg)
	client.send(bytes(msg, "utf8"))
	msg = client.recv(4096).decode("utf8")
	msg = json.loads(msg)
	print("Got message from client %s:\n %s" % (address,msg))
	client.send(bytes("disconnect","utf8"))
	msg = client.recv(4096).decode("utf8")
	if msg == "ok":
		print("Server finished to handle client %s:" % address)
	else:
		print("Got message from client %s:\n %s" % (address,msg))
	client.close()
	
	

PORT = 5555
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost",PORT))
server.listen(10)
while True:
	client, client_address = server.accept()
	print("%s:%s has connected." % client_address)
	Thread(target=handle_client, args=(client,client_address)).start()

server.close()


