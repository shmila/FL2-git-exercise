import csv
import numpy as np



def loss_func(y,y_tag):
    assert y.size==y_tag.size
    return np.mean(np.square(y-y_tag),axis=0)

def derative(f,h=1e-8):
    def func(x):
        ders=np.zeros_like(x)
        for i in range(x.size):
            ders[i] = (f(x[i]+h)-f(x[i]))/h
        return ders
    return func

def gd(f,x0,max_iter=10000,lr=0.5):
    x=x0
    der = derative(f)
    counter = 0
    while counter<max_iter:
        curr_der = der(x)
        if curr_der == 0:
            break
        x=x-lr*curr_der
        counter +=1
    return x

f = lambda x: np.square(x[0]-5) + np.square(x[1]+2)
x0 = np.zeros(2)
min_x,min_y = gd(f,x0)
a=9





# path = 'D3.csv'
# fs = csv.reader(open(path))
# all_rows = []
# for r in fs:
#     all_rows.append(r)
# data = np.zeros((len(all_rows),len(all_rows[0])))
# for i in range(len(all_rows)):
#     dr = [float(x) for x in all_rows[i]]
#     data[i] = dr
# x = data[:,0]
# y= data[:,1]
