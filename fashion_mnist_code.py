from utils import mnist_reader
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

def loss_grads(X, y, W, reg):
    # x = (m,d) w = (k,d) , y = (m), m=batch size
    m = y.shape[0]
    k, d = W.shape
    s = X.dot(W.T) # (m,k)
    m0 =np.max(s,axis=1)[:,np.newaxis]
    s = s-m0
    exps = np.exp(s) # (m,k)
    sums = np.sum(exps,axis=1) # m

    correct_exps = np.array([exps[i,y_] for i,y_ in enumerate(y)]) # m
    eps = 1e-8
    hs = (correct_exps/sums)
    losses = -np.log(hs+eps)
    base_grad = X * (hs)[:, np.newaxis] # m,d
    grads = np.zeros((m, k, d))
    for mm in range(m):
        grads[mm]=np.matlib.repmat(base_grad[mm],k,1)
        # adjust the correct grad in classifier yj
        correct_class = y[mm]
        grads[mm,correct_class]-= X[mm, :]

    #add regularization
    loss = np.mean(losses)
    loss += 0.5*reg*np.sum(W*W)

    grad = np.mean(grads,axis=0)
    grad += reg*W

    return loss,grad

def train(X,y,batch_size=32,num_iter=100,reg=1e3,lr=1e-4):
    num_ex, d = X.shape
    k = np.unique(y).size
    np.random.seed(124)
    W = np.random.randn(k, d)

    loss_history=[]
    for i in range(num_iter):
        idxs = np.random.choice(num_ex,batch_size,replace=False)
        loss,grad = loss_grads(X[idxs,:],y[idxs],W,reg)
        loss_history.append(loss)
        #update the weights
        W -= lr*grad

    return loss_history, W

def predict(X,W):
    pred_y = np.zeros_like(X.shape[0])
    scores = X.dot(W.T)
    pred_y = np.argmax(scores,axis = 1)
    return pred_y



X_train, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

print 'example shape is: ',X_train.shape
print 'label shape is: ',y_train.shape

print 'example shape is: ',X_test.shape
print 'label shape is: ',y_test.shape

num_ex,d = X_train.shape
x_train = np.ones((num_ex,d+1))*255
x_train[:,:-1] = X_train
x_train /= 255.
num_ex,d = X_test.shape
x_test = np.ones((num_ex,d+1))*255
x_test[:,:-1] = X_test
x_test /= 255.


# k=np.unique(y_train).size
# np.random.seed(0)
# W = np.random.randn(k,d+1)

loss_hist , W = train(x_train,y_train,batch_size=128,num_iter=10000,lr=1e-4,reg=1e4)
loss_hist = np.array(loss_hist)
print('last loss value is: '+str(loss_hist[-1]))

pred_y_train = predict(x_train,W)
print('the train accuracy is: %.2f' % np.mean(y_train==pred_y_train))


pred_y_test = predict(x_test,W)

print('the test accuracy is: %.2f' % np.mean(y_test==pred_y_test))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(W[i,:-1].reshape(28,28),cmap='gray')

plt.figure()
plt.plot(loss_hist)
plt.show()





# train_batch(x_train[:batch_size, :], y_train[:batch_size], W)
# loss_grads(np.ones((2, 3)), np.array(range(1, 3)), np.random.randn(4, 3),1e3)

