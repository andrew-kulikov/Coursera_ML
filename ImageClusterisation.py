from skimage.io import imread
from skimage import img_as_float
import pylab
import numpy as np
from sklearn.cluster import KMeans
import math


def GetImage(imshape, pixels):
    im = []
    row = []
    j = 0
    for i in range(len(pixels)):
        if j < imshape[1]:
            row.append(pixels[i])
            j += 1
        else:
            im.append(np.array(row))
            row = [pixels[i]]
            j = 1
    im.append(np.array(row))
    return np.array(im)

def PSNR(im1, im2):
    if im1.shape != im2.shape:
        return -1
    m, n = im1.shape[:2]
    MSE = 0
    for i in range(m):
        for j in range(n):
            for k in range(3):
                MSE += abs(im1[i][j][k] - im2[i][j][k])**2
    MSE /= 3 * n * m
    return 20 * math.log10(1 / math.sqrt(MSE))

def ImageToArray(image):
    X = []
    for i in range(len(image)):
        for j in range(len(image[i])):
            X.append(image[i][j])
    return np.array(X)

def TransformColors(X, labels, n_clusters, rule='average'):
    X1 = sorted(list(zip(labels, X)), key=lambda t: t[0])
    colors = np.zeros((n_clusters, 3))
    labels1 = np.zeros(len(labels))
    
    for i in range(len(labels)):
        labels1[i] = labels[i]
    labels1.sort()
    
    X1 = np.array([t[1] for t in X1])
    
    for i in range(n_clusters):
        positions = np.where(labels1 == i)
        cur = X1[positions[0][0]:positions[0][-1] + 1]
        for j in range(3):
            cluster_colors = [t[j] for t in cur]
            if rule == 'average':
                colors[i][j] = np.average(cluster_colors)
            elif rule == 'median':
                colors[i][j] = np.median(cluster_colors)
    
    for i in range(len(X)):
        X[i] = colors[labels[i]]
    
    return X      

def GetTrainedImage(X, km, shape, rule='average'):
    X = TransformColors(X, km.labels_, km.n_clusters, rule=rule)
    image1 = GetImage(shape, X)
    pylab.imshow(image1)
    pylab.show()
    return image1

def main():
    image = imread('Data/parrots.jpg')
    image = img_as_float(image)
    
    pylab.imshow(image)
    pylab.show()
    
    log = open('ImageLog.txt', 'w')
    ans = open('ImageClussterizer.txt', 'w')
    
    for n_clusters in range(1, 21):
        km = KMeans(n_clusters=n_clusters, random_state=241)
        X = ImageToArray(image)
        km.fit(X)
        
        psnr_av = PSNR(image, GetTrainedImage(X, km, image.shape))
        psnr_med = PSNR(image, GetTrainedImage(X, km, image.shape, rule='median'))
        print('Step #',
              n_clusters,
              ': average PSNR = ',
              psnr_av,
              '. Median PSNR = ',
              psnr_med,
              file=log)
        log.flush()
        if max(psnr_av, psnr_med) > 20:
            print('-----------------------\n',
                  'Training was sucsessfuly ended on the iteration #',
                  n_clusters,
                  file=log)
            print(n_clusters, end='', file=ans)
            log.close()
            ans.close()
            break

    log.close()
    ans.close()

if __name__ == '__main__':
    main()