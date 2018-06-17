import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt

def svd_recover_img(T,k):
    """

    :param img: a matrix
    :param k: number of greatest values to choose from the Eigenvalues
    :return: the new img
    """

    u,sigma,v_t = np.linalg.svd(T)
    #need to handle zeros

    sigma_inverse = 1/sigma

    sigma_inverse = sigma_inverse[::-1]
    sigma_inverse[np.isinf(sigma_inverse)] = 0
    sigma_inverse[np.isnan(sigma_inverse)] = 0


    #now adjust the sizes using the k rows
    new_sigma_matrix = np.diag(sigma_inverse[:k])
    new_v = v_t.T[::-1,::-1]
    new_v = new_v[:,:k]
    new_u = u[::-1,::-1].T
    new_u = new_u[:k,:]
    new_m_t = np.matmul(np.matmul(new_v,new_sigma_matrix),new_u)

    return new_m_t.T


with open('rainbow.pickle','rb') as img_file:
    a=img_file.read()
    img = pickle.loads(a)[:,:,0]/255.0
# plt.show()
rows,cols = img.shape
precent = 0.5
new_rows = int(rows*precent)
new_cols = int(cols*precent)
tiny_img = copy.deepcopy(img[:new_rows,:new_cols])
plt.subplot(131)
plt.imshow(tiny_img,cmap='gray')

T = np.eye(new_rows,new_rows)#+0.01*np.random.randn(new_rows,new_rows)

img_dec = np.matmul(T,tiny_img)
plt.subplot(132)
plt.imshow(img_dec, cmap='gray')

rec_T = svd_recover_img(T, 500)
new_img = np.matmul(rec_T, img_dec)

plt.subplot(133)
plt.imshow(new_img, cmap='gray')
plt.show()



