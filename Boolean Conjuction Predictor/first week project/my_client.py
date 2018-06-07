# client.py
import random
import socket
import matplotlib.pyplot as plt

test_set = []
training_set = []


def image_show(img):
    img.reshape((28, 28))
    imgplot = plt.imshow(img)
    plt.show()


def main_loop(training_set, test_set):
    k = 0.6
    # create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # get local machine name
    host = socket.gethostname()

    port = 4444
    images_path = "filtered mnist.txt"
    # test_set = create_test_set()
    # connection to hostname on the port.
    s.connect((host, port))

    # Receive no more than 1024 bytes
    # s.send(images_path)
    # send training set to server
    s.send(str(k))
    # recieve ready bit
    # tm = s.recv(1024)
    # send training set to server
    s.recv(1024)
    s.send(training_set)
    s.recv(1024)
    ready_bit = s.recv(1024)
    if ready_bit == 1:
        for test in test_set:
            image_show(test)
        s.send(test_set)
    s.close()


def label_data():
    with open("filtered mnist.txt", 'r+') as data:
        images = data.readlines()
        with open("labeled_data.txt", 'w+') as labeled:
            for i in range(10):
                for j in range(100):
                    labeled.write(images[i * 100 + j] + ' ' + str(i) + "\n")


def create_training_set_and_test_set():
    test_set = []
    training_set = []
    with open("labeled_data.txt", 'r+') as data:
        images = data.readlines()
        for i in range(10):
            # test_set += random.sample(images[100 * i:100 * i + 100], 10)
            training_set += random.sample(images[100 * i:100 * i + 100], 90)
        test_set += [image for image in images if image not in training_set]
    with open("training_set.txt", 'w+') as training_file:
        for line in training_set:
            training_file.write(line + str('\n'))
    with open("test_set.txt", 'w+') as test_file:
        for line in test_set:
            test_file.write(line[:-2] + str('\n'))
    return ('training_set.txt', 'test_set.txt')


def main():
    label_data()
    (training_set_path, test_set_path) = create_training_set_and_test_set()
    main_loop(test_set_path, training_set_path)


if __name__ == "__main__":
    main()
