import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy

def k_means(k,img,max_iter):
    rows,cols,num_colors = img.shape
    new_centroids = init_centroids(k,img)
    centroids = np.zeros_like(new_centroids)
    counter=0
    while (centroids!=new_centroids).all() and counter<max_iter:
        table = (centroids!=new_centroids)
        centroids = new_centroids
        # calculate dists for each centroid
        dists = calc_dist(k,img,centroids)
        # assign each pixel to its centroid
        match_k_index = get_centroid_match(dists,rows,cols)
        # calculate the new centroids
        new_centroids = calc_new_centroids(k,num_colors,match_k_index)
        counter += 1
    return new_centroids

def init_centroids(k,img):
    rows, cols, num_colors = img.shape
    rand_r = np.random.randint(0,rows-1,k)
    rand_c = np.random.randint(0,cols-1,k)
    centroids = copy.deepcopy(img[rand_r,rand_c,:])
    # centroids = np.random.uniform(0,1,(k, num_colors))
    return centroids

def calc_dist(k,img,centroids):
    rows,cols,num_colors = img.shape
    dists = np.zeros((k,rows,cols))
    for i in range(k):
        dists[i] = np.sqrt(np.sum(np.square(img - centroids[i]), axis=2))
    return dists
def get_centroid_match(dists,rows,cols):
    # assign each pixel to its centroid
    match_k_index = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            vec = dists[:, r, c]
            match_k_index[r, c] = np.argmin(vec)
    print('active centroids: %s' % np.unique(match_k_index))
    return match_k_index
def calc_new_centroids(k,num_colors,match_k_index):
    centroids = np.zeros((k,num_colors))
    # calculate the new centroids
    for kk in range(k):
        ind = np.where(match_k_index == kk)
        if ind[0].size == 0:
            continue
        avg = np.zeros(num_colors, dtype=float)
        for c in range(num_colors):
            cl = img[ind][:, c]
            avg[c] = np.mean(cl)
        centroids[kk] = avg
    return centroids



with open('rainbow.pickle','rb') as img_file:
    a=img_file.read()
    img = pickle.loads(a)
img = img[:,:,::-1]

naive_compr_img = (img / 16) *16
plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.title('compressed')
plt.imshow(naive_compr_img)
img = img/255.0
k = 16
tiny_img = img
rows,cols,num_color = tiny_img.shape

centroids = k_means(k,tiny_img,20)
new_img = np.zeros_like(tiny_img)
# calculate dists for each centroid
dists = calc_dist(k,img,centroids)
# assign each pixel to its centroid
match_k_index = get_centroid_match(dists,rows,cols)
new_colors = (centroids*255).astype(int)
for r in range(rows):
    for c in range(cols):
        color = new_colors[int(match_k_index[r, c])]
        new_img[r,c] = color

plt.subplot(133)
plt.title('k-means k=%i'%k)
plt.imshow(new_img)
plt.show()


