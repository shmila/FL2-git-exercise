# server.py
import random
import socket
import time
import traceback

from Classifier import Classifier


def send_to_client(clientsocket, message):
    try:
        clientsocket.send(message)
    except:
        clientsocket.close()
        print("client " + str(clientsocket) + "quit")
        clientsocket.remove(clientsocket)


def build_classifiers(training_set_path):
    labled_images = []
    with open(training_set_path, 'r+') as images:
        for i in range(10):
            labled_images.append([])
        labled_images.append([image for image in images if image.endswith('0')])
        labled_images.append([image for image in images if image.endswith('1')])
        labled_images.append([image for image in images if image.endswith('2')])
        labled_images.append([image for image in images if image.endswith('3')])
        labled_images.append([image for image in images if image.endswith('4')])
        labled_images.append([image for image in images if image.endswith('5')])
        labled_images.append([image for image in images if image.endswith('6')])
        labled_images.append([image for image in images if image.endswith('7')])
        labled_images.append([image for image in images if image.endswith('8')])
        labled_images.append([image for image in images if image.endswith('9')])
        # labled_images[0] = [image for image in images if image.endswith('0')]
        # labled_images[1] = [image for image in images if image.endswith('1')]
        # labled_images[2] = [image for image in images if image.endswith('2')]
        # labled_images[3] = [image for image in images if image.endswith('3')]
        # labled_images[4] = [image for image in images if image.endswith('4')]
        # labled_images[5] = [image for image in images if image.endswith('5')]
        # labled_images[6] = [image for image in images if image.endswith('6')]
        # labled_images[7] = [image for image in images if image.endswith('7')]
        # labled_images[8] = [image for image in images if image.endswith('8')]
        # labled_images[9] = [image for image in images if image.endswith('9')]
        # images_of_other += (images_of_one + images_of_two + images_of_three + images_of_four + images_of_five +
    #                     images_of_six + images_of_seven + images_of_eight + images_of_nine)
    print('labled_images is:' + str(labled_images) + '. labeled_images len is: ' + str(len(labled_images)))
    classifiers = []
    positive_images = []
    for i in range(10):
        positive_images.append(labled_images[i])
        negative_images = []
        for j in range(10):
            if j != i:
                negative_images.append(labled_images[j])
        curr_classifier = Classifier.__init__(positive_images, negative_images)
        print('built classifier ' + str(curr_classifier))
        classifiers.append(curr_classifier)
    hypotheses = []
    for classifier in classifiers:
        images = classifier.label_correctly()
        print('images: ' + str(images))
        hypotheses.append(classifier.train(images))
        return hypotheses


# def get_evaluated_digit():
#     with open(test_set_path, 'r+') as images:
#         ##TODO delete labeling
#         for image in range(images):
#             h = Classifier.train(image)
#             image_evaluation = []
#             for i in range(10):
#                 hypoth

def test_example(classifier, t):
    return classifier.evaluate(t)


def test(classifiers, test_path):
    image_dictionary = {}
    with open(test_path, 'r+') as test_images:
        for image in test_images:
            found = False
            for i in range(10):
                if test_example(classifiers[i], image) == 1:
                    image_dictionary[image] = i
                    found = True
                    break
            if found is False:
                image_dictionary[image] = random.randint(0, 9)
    return image_dictionary


def main_loop():
    # create a socket object
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # get local machine name
    host = socket.gethostname()

    port = 4444

    # bind to the port
    serversocket.bind((host, port))

    # queue up to 5 requests
    serversocket.listen(5)

    # establish a connection
    client_socket, addr = serversocket.accept()
    print("Got a connection from %s" % str(addr))
    # currentTime = time.ctime(time.time()) + "\r\n"
    try:
        data = client_socket.recv(1024)
        if data == "":
            print("quit1!")
            client_socket.close()
        else:
            k = float(data)
            send_to_client(client_socket, 'got k')
        data = client_socket.recv(1024)
        if data == "":
            print("quit2!")
            client_socket.close()
        else:
            training_set_path = data
            classifiers = build_classifiers(training_set_path)
            send_to_client(client_socket, '1')

        data = client_socket.recv(1024)
        send_to_client(client_socket, 'got training set')
        if data == "":
            print("quit3!")
            client_socket.close()
        else:
            test_path = data
            image_dictionary = test(classifiers, test_path)
            with open('images_labeling_output.txt', 'r+') as output:
                for image in image_dictionary:
                    output.write(image_dictionary.get(image))
                send_to_client(client_socket, 'images_labeling_output.txt')

    except:
        tb = traceback.format_exc()
        print(tb)
        client_socket.close()

    client_socket.close()


def main():
    main_loop()


if __name__ == "__main__":
    main()
