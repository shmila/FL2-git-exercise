import csv
import numpy as np



def loss_func(y,y_tag):
    assert y.size==y_tag.size
    return np.mean(np.square(y-y_tag),axis=0)

def derative(f,h=1e-8):
    def func(x):
        ders=np.zeros_like(x)
        for i in range(x.size):
            # make all but the current x component zero
            new_x = np.zeros_like(x)
            new_x_h = np.zeros_like(x)
            new_x[i] = x[i]
            new_x_h[i] = x[i] + h
            ders[i] = (f(new_x_h)-f(new_x))/h
        return ders
    return func

def gd(f,x0,max_iter=100000,lr=1e-4):
    x=x0
    der = derative(f)
    counter = 0
    while counter<max_iter:
        curr_der = der(x)
        # if curr_der.all() == 0:
        #     break
        x=x-lr*curr_der
        counter +=1
    return x

f = lambda x: np.square(x[0]-5) + np.square(x[1]+2)
x0 = np.zeros(2)
# x0 = np.array([2,-1])
min_x,min_y = gd(f,x0)
print('for first function the min values are: x=%.2f y=%.2f' % (min_x,min_y))
f2 = lambda x:np.square(1-(x[1]-4))+35*np.square((x[0]+6)-np.square(x[1]-4))
x0 = np.zeros(2)
# x0 = np.array([-5,5])
min_x,min_y = gd(f2,x0)
print('for second function the min values are: x=%.2f y=%.2f' % (min_x,min_y))




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
