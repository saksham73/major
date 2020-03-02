import os
import numpy as np
from scipy.misc import imread, imresize

count = -1
train_labels = []
train_images = []


for root, dirs, files in os.walk(os.getcwd()):
    for folder in dirs:
        if(folder!='.ipynb_checkpoints'):
            count = count + 1
            files = os.listdir(folder)
            print("Class " + str(count) + " - " + str(len(files)))
            for file in files:
                img = imresize(imread(os.getcwd()+'/'+folder+'/'+file, mode='RGB'), (60, 60)).astype(np.float32)
                #img[:, :, 0] -= 123.68
                #img[:, :, 1] -= 116.779
                #img[:, :, 2] -= 103.939
                #img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
                #img = img.transpose((2, 0, 1))
                #img = np.expand_dims(img, axis=0)
                train_images.append(img)
                train_labels.append(count)

print(len(train_images))
print(len(train_labels))
print("Total no of classes - " + (count + 1))
#print(train_labels)
np.save('train_all_images_lenet_60.npy',np.array(train_images))
np.save('train_all_labels_lenet_60.npy',np.array(train_labels))