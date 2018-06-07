import numpy as np
import socket as skt


class Client(object):
    def __init__(self):
        '''
        :param suit: defult = 1 within 1-4
        :param number: defult = 1 within  1-13
        # defining suits
        # 4 = clubs (tiltan)
        # 3 = spades (Alim)
        # 2 = hearts
        # 1 = diamonds
        '''
        self.name = 'client'
        self.socket = None

        self.raw_dataX = []
        self.raw_dataY = []

        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []
        self.testIdx = []

        self.trainAddX = ''
        self.trainAddY = ''
        self.testAdd = ''

        self.kAccDict = dict()

    def loadData(self,input_file):
        x = np.loadtxt(input_file)
        dig_list = range(10)
        y = np.repeat(dig_list, 100)
        self.raw_dataX = x
        self.raw_dataY = y

        return (x, y)

    def splitTest(self, x = None, y = None, numTest=10):
        """
        spliting out 10% of each digit to test
        :param x: images (byt column)
        :param y: digit tag
        :return:
        """
        if x == None:
            x = self.raw_dataX
        if y == None:
            y = self.raw_dataY
        test_idx = []  # list of indices to go to test
        for dig in range(10):
            for imNum in range(numTest):  # take numTest (10) instances from each digit
                idx = np.random.randint(0, 100) + dig * 100  # gives out idx between 0-99 then 100-199 then 200-299 ...
                while idx in test_idx:
                    idx = np.random.randint(0,
                                            100) + dig * 100  # gives out idx between 0-99 then 100-199 then 200-299 ...
                test_idx.append(idx)

        # idxArr = np.array(test_idx)
        self.testY = y[test_idx]
        self.testX = x[test_idx, :]
        neg_mask = np.ones(1000)  # create a negative mask where indices will be false and others true
        neg_mask[test_idx] = 0  # choose indices to be false (those taken to test)
        bool_mask = np.array(neg_mask, dtype=bool)
        self.trainY = y[bool_mask]
        self.trainX = x[bool_mask, :]
        self.testIdx = test_idx

        return [self.trainX, self.trainY, self.testX, self.testY, self.testIdx]


    def createLink(self, x = None):

        if x!= None:
            np.savetxt("X.txt", self.trainX, fmt='%1d')
            self.trainAddX = 'X.txt'
        else:

            np.savetxt("trainX.txt", self.trainX, fmt='%1d')
            np.savetxt("trainY.txt", self.trainY, fmt='%1d')
            np.savetxt("testX.txt", self.trainY, fmt='%1d')

            self.trainAddX = 'trainX.txt'
            self.trainAddY = 'trainY.txt'
            self.testAdd = 'testX.txt'

        return

    def sendTrain(self):
        # send link
        return

    def evalPredict(self, predY, testY = None):
        if testY == None:
            testY = self.testY

        # predY = np.load(predAdd)

        accuracy = np.sum(testY == predY)/100  ### Check formats

        return accuracy

    def estConn(self, host_name = 'localhost', port = 5555):
        buff_size = 4096
        self.socket = skt.socket(skt.AF_INET, skt.SOCK_STREAM)
        self.socket.connect((host_name, port))
        msg = self.socket.recv(buff_size)
        print(msg)


    def evalK(self, k):
        buff_size = 4096
        # train
        self.socket.send(str(k) +self.trainAddX + self.trainAddY)
        msg = self.socket.recv(buff_size)
        print("msg")

        # test
        predY = []
        for image in self.testX:
            imInt = image.astype(int)
            imStr = np.array2string(imInt)
            imStr = imStr[1:-1]
            self.socket.send(imStr)  # send one image
            tag = int(self.socket.recv(buff_size))  # recieve its eval dig
            predY.append(tag)

        self.socket.send('stop')

        predY = np.array(predY) # cast prediction as a nparray

        kSucc = self.evalPredict(predY)  # recive kSucc
        self.kAccDict[k] = kSucc
        print('k success perc is: ' + str(kSucc))


if __name__ == "__main__":
    c = Client()
    c.loadData("filtered mnist")
    c.splitTest()
    for image in c.testX:
        imageInt = image.astype(int)
        istr = np.array2string(imageInt)
        istr = istr[1:-1]
        print(type(istr))
        print(len(istr))
        print(istr)
        break
    # c.createLink()
    # c.estConn()
    # for k in [0.5, 0.1, 0.15, 0.2]
    #     c

        




