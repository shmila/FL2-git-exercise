import socket
import threading
import numpy as np
import mnist

class Server:

    def __init__(self,port=5552):
        self.host = 'localhost'
        self.port = port
        # create a socket object
        self.server = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.clients = []
        self.hs = {}

    def start(self):
        self.server.bind((self.host,self.port))
        self.server.listen(2)
        while True:
            # establish a connection
            client, addr = self.server.accept()
            self.clients.append(client)

            print("Got a connection from %s" % str(addr))

            threading.Thread(target=self.handle_client, args=(client, addr)).start()

    def convert_str_to_x(self,st_x):
        st_x = st_x.strip()
        return np.fromstring(st_x, dtype=int, sep=' ')



    def handle_client(self,client,addr):
        buff_size = 4096
        while True:
            client.send("Server is waiting for training data")
            msg = client.recv(buff_size)
            if msg == 'quit':
                break
            #now we got path to data file
            print("Got %s" % msg)
            args = msg.split(',')
            assert len(args)==3
            k = float(args[0])
            x_path = args[1]
            y_path = args[2]
            X_train = np.loadtxt(x_path)
            y_train = np.loadtxt(y_path)
            m = mnist.MnistClassifier()
            m.train(X_train,y_train,k)
            #finish training
            print("Finish to train on data")
            client.send("Ready for Testing!")
            msg = client.recv(buff_size)
            print("Got %s" % msg)
            while msg.find('1') != -1 or msg.find('0') != -1:
                #we got 1 test example
                x = self.convert_str_to_x(msg)
                res = m.pred(x)
                client.send(str(res))
                msg = client.recv(buff_size)
                print("Got %s" % msg)
        client.close()
        self.clients.remove(client)
        print('Finish to handle client')


if __name__ == "__main__":
    server = Server()
    server.start()






