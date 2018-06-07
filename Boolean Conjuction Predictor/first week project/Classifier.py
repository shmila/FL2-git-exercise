class Classifier:
    def __init__(self, images_pos, images_neg):
        # Each classifier has a set of examples
        self.images_positive = images_pos
        self.images_negavtive = images_neg
        self.hypothesis = []

    def evaluate(self, t):
        res = 1
        for i in range(len(self.h)):
            if (self.h[i] == 1):
                if (i % 2 == 0):
                    # temp = int(t[i / 2])
                    res = res * (t[i / 2])
                else:
                    # temp = int(1 - t[i / 2])
                    res = res * (1 - t[i / 2])
        return res

    def label_correctly(self):
        images = []
        print('in method label correctly, len of images_positive is: ' + str(len(self.images_positive)))
        print('in method label correctly, len of images_negavtive is: ' + str(len(self.images_negavtive)))
        for image in self.images_positive:
            print(image)
            print('len of image ' + str(len(image)))
            image[-1] = '1'
        for image in self.images_negavtive:
            image[-1] = '0'
        images += self.images_positive
        images += self.images_negavtive
        print('in method label correctly, len of images_negavtive is: ' + str(len(self.images_negavtive)))
        return images

    def all_negative_hypothesis(self, n):
        return [1 for i in range(2 * n)]

    def train(self, images):
        # Determine the conjunction function h, according to the images (i.e. examples) and the given label (i.e. i)
        print ('images' + str(images))
        n = len(images[0]) - 1
        h = self.all_negative_hypothesis(n)
        for t in images:
            predicted_value = self.evaluate(h, t)
            actual_value = t[-1]
            if actual_value == 1 and predicted_value == 0:
                for i in range(n):
                    if t[i] == 1:
                        h[2 * i + 1] = 0
                    else:
                        h[2 * i] = 0
            if actual_value == 0:
                continue
        self.hypothesis = h
        # return h
