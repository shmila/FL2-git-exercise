#redis server
import rwlock 
import threading
import socket
import signal
import sys


MSG_NEW_REQUEST = "Command options: 'SET', 'GET', 'SHOW_ALL', 'CLIENTS', 'QUIT'\n"
commands_descriptions = {'SET':'insert key-value and this will update or create new entity in the db', \
						'GET':'insert key to get its value', 'SHOW_ALL':'show all the database',\
						'CLIENTS':'get the current clients connected to db', \
						'QUIT':'finish the connection to db'}
						
def handle_command(client,com):
	resp = ''
	if com == 'SET':
		client.send(bytes(commands_descriptions['SET']+'\n','utf8'))
		msg = client.recv(SIZE_MSG).decode('utf8')
		com_parts = msg.split('-')
		if len(com_parts) == 2:
			rwl.acquire_write()
			db[com_parts[0]] = com_parts[1]
			resp = 'Succeed'
			rwl.release()
		else:
			resp = 'illegal user input'
	elif com == 'GET':
		client.send(bytes(commands_descriptions['GET']+'\n','utf8'))
		msg = client.recv(SIZE_MSG).decode('utf8')
		rwl.acquire_read()
		try:
			res = db[msg]
			resp = res
		except:
			resp = 'operation failed'
		rwl.release()
	elif com == 'SHOW_ALL':
		for key,val in db.items():
			resp += key+'-'+val+', '
		resp = resp[:-2]
	elif com == 'CLIENTS':
		#maybe i need to put a lock here also so the info will be valid
		for c in clients:
			resp += str(c)+', '
		resp = resp[:-2]
	return resp
	
def handle_client(client,client_address):
	clients.append(client_address)
	address = client_address[0]
	msg = MSG_NEW_REQUEST
	client.send(bytes(msg,'utf8'))
	while True:
		msg = client.recv(SIZE_MSG).decode('utf8')
		msg = msg.upper()
		if msg == 'QUIT':
			client.send(bytes('bye bye','utf8'))
			break
		else:
			try:
				res = handle_command(client,msg) + '\n' + MSG_NEW_REQUEST
				client.send(bytes(res,'utf8'))
			except:
				res = 'Failed!\n'+ MSG_NEW_REQUEST
				client.send(bytes(res,'utf8'))
				continue			
	# finish to handle the client
	print('Finish handling client ',client_address)
	clients.remove(client_address)
	
def signal_handler(signal, frame):
	#close server when user press ctrl+c
	global flag
	flag = False
	raise
	
	
SIZE_MSG = 4096
PORT = 5555
clients = []
db = {}
rwl = rwlock.RWLock()
print('Starting server')
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost",PORT))
server.listen(10)
flag = True
signal.signal(signal.SIGINT, signal_handler)
while flag:
	try:
		client, client_address = server.accept()
		print("%s:%s has connected." % client_address)
		threading.Thread(target=handle_client, args=(client,client_address)).start()
	except:
		pass
print('Closing server')
server.close()


