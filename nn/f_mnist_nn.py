from utils import mnist_reader
import numpy as np
import matplotlib.pyplot as plt
from fc_network import *
from sklearn.model_selection import train_test_split

seed = 1


X_train, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

x_train, x_val, yy_train, yy_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
# x_train = x_train.astype(float) / 255.
# x_val = x_val.astype(float) / 255.

num_ex,d = X_train.shape
k = np.unique(y_train).size

lr=1e-4
ws = 1e-1
model = FullyConnectedNet([100],input_dim=d,num_classes=k,seed=seed,weight_scale=ws,reg=0.0,active_func='relu')
data = {
    'X_train': x_train,
    'y_train': yy_train,
    'X_val': x_val,
    'y_val': yy_val,
}
solver = Solver(model,data,print_every=100,num_epochs=20,batch_size=128,lr=lr,verbose=True,lr_decay=1)
solver.train()

acc=solver.check_accuracy(X_test,y_test)
print 'accuracy is:',acc
plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()

# plt.figure()
# #sample some w from first layer
# ws = model.get_weights()
# num_pics_per_layer = 10
# w = ws[0]
# indx = np.random.choice(w.shape[1],num_pics_per_layer)
# for j in range(num_pics_per_layer):
#     pos = j
#     plt.subplot(1,num_pics_per_layer,pos+1)
#     pic = w[:,indx[j]]
#     plt.imshow(pic.reshape(28,28),cmap='gray')
#
# plt.show()