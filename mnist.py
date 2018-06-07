import numpy as np
import random
import copy
import sys
import matplotlib
from matplotlib import pyplot as plt


class MnistClassifier:

    def __init__(self):

        self.hs = {}


    def consistency_algorithm(self,X, y, min_prob=1):
        n = X.shape[1]
        h = self.all_negative_hypothesis(n)
        for k in range(X.shape[0]):
            t = X[k]
            predicted_value = self.evaluate(h, t, min_prob)
            actual_value = y[k]
            if actual_value == 1 and predicted_value == 0:
                for i in range(n):
                    if t[i] == 1:
                        h[2 * i + 1] = 0
                    else:
                        h[2 * i] = 0
            if actual_value == 0:
                continue
        return h

    def evaluate(self,h, t, min_prob=1):
        # the number of errors
        num_err = 0.0
        total_ones = h.count(1)
        if total_ones == 0:
            return 1
        for i in range(len(h)):
            if (h[i] == 1):
                if (i % 2 == 0):
                    if t[i / 2] == 0:
                        if min_prob == 1:
                            return 0
                        else:
                            num_err += 1
                elif (1 - t[i / 2]) == 0:
                    if min_prob == 1:
                        return 0
                    else:
                        num_err += 1
        prob = num_err / total_ones
        if 1 - prob >= min_prob:
            return 1
        return 0

    def all_negative_hypothesis(self,n):
        return [1 for i in range(2 * n)]

    def eval_all(self,h, X, min_prob=1):
        rows = X.shape[0]
        res = np.zeros(rows)
        for i in range(rows):
            res[i] = self.evaluate(h, X[i], min_prob)
        return res

    def eval_h(self,X_train, y_train, X_val, y_val):
        h = self.consistency_algorithm(X_train, y_train)
        pred_y = self.eval_all(h, X_val)
        acc = np.mean(pred_y == y_val)
        return h, acc

    def train(self,X,y,k,num_of_voters=3):
        # use cross validation
        hist = {}
        #init dict
        for i in range(10):
            hist[i] = []

        for j in range(num_of_voters):
            x_train, y_train, x_val, y_val = self.separate_data(X,y,k)
            for i in range(10):
                # train each h on its examples
                ind_org = np.where(y_train == i)[0]
                x_dig = x_train[ind_org]
                y_dig = np.ones(x_dig.shape[0])
                hi = self.consistency_algorithm(x_dig,y_dig)
                pred_y = self.eval_all(hi, x_val)
                acc = np.mean(pred_y == y_val)
                # hist[i].append((hi,acc))
                hist[i].append(hi)
        self.hs = hist

    def pred(self,x):
        # assume x is one example
        temp = self.hs[0]
        num_voters = len(self.hs[0])
        preds_prob = []
        for i in range(len(self.hs)):
            preds = []
            num_ones = 0
            num_zeros = 0
            for h in self.hs[i]:
                res = self.evaluate(h,x)
                if res == 1:
                    num_ones += 1
                else:
                    num_zeros += 1
                preds.append(res)

            if num_ones > num_zeros:
                #res is 1
                ans = 1
                prob = float(num_ones)/num_voters
            else:
                #res is 0
                ans = 0
                prob = float(num_zeros)/num_voters
            preds_prob.append((ans,prob))
        max = 0
        ind_max = 0
        for i in range(len(preds_prob)):
            p = preds_prob[i]
            if p[1]>max:
                max = p[1]
                ind_max = i
        return preds_prob[ind_max][0]










    def separate_data(self,X,y,k):
        # separate into train and validation sets
        num_examples, dim = X.shape

        num_val = int(num_examples * k)
        x_train = copy.deepcopy(X)
        y_train = copy.deepcopy(y)
        x_val = np.zeros((num_val,dim))
        y_val = np.zeros(num_val)

        num_examples_per_dig = num_val / 10

        for i in range(10):
            ind_org = np.where(y==i)[0]
            x_dig = copy.deepcopy(X[ind_org])
            y_dig = np.full(x_dig.shape[0],i)

            ind = random.sample(range(y_dig.size), num_examples_per_dig)
            start = i * num_examples_per_dig
            end = start + num_examples_per_dig
            x_val[start:end] = x_dig[ind]
            y_val[start:end] = y_dig[ind]
            x_train = np.delete(x_train,ind,axis=0)
            y_train = np.delete(y_train,ind)

        #now it is not shuffled
        return x_train , y_train,x_val,y_val

    def create_imgs(self,X,y,num_examples_per_dig=1,show = False):
        fig, axes = plt.subplots(nrows=num_examples_per_dig, ncols=10)

        for i in range(10):
            ind_org = np.where(y == i)[0]
            sample_ind = random.sample(ind_org, num_examples_per_dig)
            x_dig = X[sample_ind]
            for j in range(num_examples_per_dig):
                if num_examples_per_dig == 1:
                    axes[i].imshow(x_dig[j].reshape(28,28))
                    axes[i].axis('off')
                else:
                    axes[j,i].imshow(x_dig[j].reshape(28,28))
                    axes[j,i].axis('off')
        if show:
            # use this only in the last call of this function
            plt.show()




# if __name__ == "__main__":
#
#
#     # path = 'data.txt'
#     # examples = np.loadtxt(path)
#     # X = examples[:,:-1]
#     # y = examples[:,-1]
#     # h=consistency_algorithm(X,y)
#     # for t in examples:
#     #     res = evaluate(h,t)
#
#     # path='data2.txt'
#     # X = np.loadtxt(path)
#     # num_rows = X.shape[0]
#     # y = np.zeros(num_rows)
#     # num_per_dig = num_rows / 10
#     # for i in range(10):
#     #     start = i * num_per_dig
#     #     end = start + num_per_dig
#     #     y[start:end] = i
#     # k = 1./10
#     # x_train, y_train, x_val, y_val = separate_data(X,y,k)
#     # inds = np.where(x_val==X)
#     # d = 10
#     # lim = 1000
#     # with open(path, 'w') as the_file:
#     #     for i in range(lim):
#     #         st = bin(i)[2:].zfill(d)
#     #         new_st = ''
#     #         for c in st:
#     #             new_st+=c+' '
#     #         the_file.write(new_st)
#     #         the_file.write('\n')
#
#
#
#     path = 'filtered mnist'
#     X = np.loadtxt(path)
#     num_rows = X.shape[0]
#     y = np.zeros(num_rows)
#     num_per_dig = num_rows/10
#     for i in range(10):
#         start = i*num_per_dig
#         end = start + num_per_dig
#         y[start:end] = i
#     # create_imgs(X,y,20,True)
#     ind = []
#     for i in range(100):
#         ind.append(i * 10)
#     ex_x = X[ind]
#     ex_y = y[ind]
#     # create_imgs(ex_x, ex_y, 10, True)
#
#
#     hs = []
#     for i in range(10):
#         start = i * num_per_dig
#         end = start + num_per_dig
#         xx = X[start:end]
#         yy = np.ones(xx.shape[0])
#         hi = consistency_algorithm(xx,yy,0.98)
#         print(np.sum(eval_all(hi,xx,0.98)))
#         hs.append(hi)
#
#     xh = np.array(hs)
#     xh = copy.deepcopy(xh[:, 1::2])
#     yh = np.array(range(10))
#     create_imgs(xh,yh,1,True)
#
#
#     # #finish eval h
#     size = 100
#     # ind = random.sample(range(num_rows),size)
#     ind=[]
#     for i in range(100):
#         ind.append(i*10)
#     # ind = range(1000)
#     X_val = X[ind]
#     y_val = y[ind]
#     # create_imgs(X_val,y_val,10,True)
#
#     print(y_val)
#     count = 0
#     for h in hs:
#         pred_y = eval_all(h,X_val,0.98)
#         if count==0:
#             pred_y[pred_y==0]=-1
#             pred_y[pred_y==1]=0
#         else:
#             pred_y[pred_y==1] = count
#         acc = np.mean(pred_y==y_val)
#         print('for h%d acc: %f' % (count,acc))
#         count +=1
#     #     print('for digit %d:'%count)
#     #     h_temp = copy.deepcopy(h)
#     #     pic = np.zeros(X.shape[1])
#     #     pic =pic.astype(int)
#     #     for i in range(len(h_temp)/2-1):
#     #         if h_temp[i]==1:
#     #             pic[i] = 1
#     #         else:
#     #             pic[i] = 0
#     #     for i in range(pic.size):
#     #         sys.stdout.write(str(pic[i]))
#     #         if i%28==0:
#     #             print('\n')
#     #     fig = plt.figure()
#     #     plt.imshow(pic.reshape((28,28)))
#     #
#     #     count += 1



























