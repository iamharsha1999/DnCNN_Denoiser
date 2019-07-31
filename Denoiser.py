## =====This is a Google Colabaratory Version ======##

! wget http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz
! tar -xzvf stl10_binary.tar.gz

from __future__ import print_function

import sys
import os, sys, tarfile, errno
import numpy as np
import matplotlib.pyplot as plt

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urlliburllibqqurlliburllibqqy

try:
    from imageio import imsave
except:

    from scipy.misc import imsave

print(sys.version_info)

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = '/content/stl10_binary'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
DATA_PATH = '/content/stl10_binary/unlabeled_X.bin'

# path to the binary train file with labels
LABEL_PATH = '/home/harsha/Image_Processing/Data_Collected/STL_10/stl10_binary/test_y.bin'


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()


def save_image(image, name):
    imsave("%s.png" % name, image, format="png")


def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                          float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def save_images(images):
    print("Saving images to disk")
    i = 1
    for image in images:
        # label = labels[i]
        directory = '/content/stl10_binary/Unlabelled Images/'
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = directory + 'img' +str(i)
        print(filename)
        save_image(image, filename)
        i = i + 1


if __name__ == "__main__":
    # download data if needed
    # download_and_extract()

    # test to check if the image is read correctly
    with open(DATA_PATH) as f:
        image = read_single_image(f)
        plot_image(image)

    # test to check if the whole dataset is read correctly
    images = read_all_images(DATA_PATH)
    print(images.shape)
    #
    # labels = read_labels(LABEL_PATH)
    # print(labels.shape)

    # save images to disk
    save_images(images)
    
    
    import numpy as np
import random
import cv2
import os

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gauss_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy

def speckle(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    noisy = image + image * gauss
    return noisy


target_directory = '/content/stl10_binary/Noised_Images_64_64_3_Gauss'
source_directory = '/content/stl10_binary/Unlabelled Images'
renamed_directory = '/content/stl10_binary/Renamed_Images'

try:
  os.makedirs(renamed_directory, exist_ok=True)
  print("Renamed directory created")
except OSError as exc:
  if exc.errno == errno.EEXIST:
    pass
  
try:
  os.makedirs(target_directory, exist_ok=True)
  print("Target directory created")
except OSError as exc:
  if exc.errno == errno.EEXIST:
    pass


j = 1

for image in os.listdir(source_directory):
    pic = cv2.imread(source_directory + '/' + image)
    reimg = cv2.resize(pic, (64, 64))
    cv2.imwrite(renamed_directory + '/image' + str(j) + '.jpg', reimg)
    reimg = gauss_noise(reimg)
    cv2.imwrite(target_directory + '/image' + str(j) + '.jpg', reimg)
    j+=1
    
input_file_path = '/content/stl10_binary/Noised_Images_64_64_3_Gauss'
output_file_path = '/content/stl10_binary/Renamed_Images'

for image in os.listdir(input_file_path):
  img = cv2.imread(input_file_path + '/' + image,cv2.IMREAD_COLOR)
#   if img.shape[-1] == 4:
#     img = image[..., :3]
  img = cv2.resize(img,(int(64),int(64)))
  input_images.append(np.array(img))
input_images = np.array(input_images)
# input_images = input_images/255
# data = data.reshape((data.shape[0],data.shape[1],data.shape[2],1))

print("Input images created")

for image in os.listdir(output_file_path):
    img = cv2.imread(output_file_path + '/' + image, cv2.IMREAD_COLOR)
#     if img.shape[-1] == 4:
#       img = image[..., :3]
    img = cv2.resize(img,(int(64),int(64)))
    output_images.append(np.array(img))    
output_images = np.array(output_images)
# output_images = output_images/255

print("Output images created")
import cv2
import numpy as np
import os
​
input_images = []
output_images = []
​
input_file_path = '/content/stl10_binary/Noised_Images_64_64_3_Gauss'
output_file_path = '/content/stl10_binary/Renamed_Images'
​
for image in os.listdir(input_file_path):
  img = cv2.imread(input_file_path + '/' + image,cv2.IMREAD_COLOR)
#   if img.shape[-1] == 4:
#     img = image[..., :3]
  img = cv2.resize(img,(int(64),int(64)))
  input_images.append(np.array(img))
input_images = np.array(input_images)
# input_images = input_images/255
# data = data.reshape((data.shape[0],data.shape[1],data.shape[2],1))
​
print("Input images created")
​
for image in os.listdir(output_file_path):
    img = cv2.imread(output_file_path + '/' + image, cv2.IMREAD_COLOR)
#     if img.shape[-1] == 4:
#       img = image[..., :3]
    img = cv2.resize(img,(int(64),int(64)))
    output_images.append(np.array(img))    
output_images = np.array(output_images)
# output_images = output_images/255
​
print("Output images created")

[ ]
from sklearn.model_selection import train_test_split
​
train_X,valid_X,train_y,valid_y = train_test_split(input_images,output_images,test_size=0.3,random_state=13)
print("Data Ready")
​

[ ]
import numpy as np
from keras.models import Sequential
from keras.layers import  Conv2D, Deconv2D, Dense, MaxPooling2D, UpSampling2D, Input, Subtract
from keras.layers.normalization import  BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import os
​
def denoiser_model(input_image):
  
    
#     layer1 = Conv2D(3,(4,4),padding = 'same')(input_image)
#     layer2 = Conv2D(64,(4,4),padding = 'same')(layer1)
#     layer3 = MaxPooling2D((2,2))(layer2)
#     layer4 = Conv2D(64,(4,4),padding = 'same')(layer3)
#     layer5 = MaxPooling2D((2,2))(layer4)
#     layer6 = Conv2D(128,(4,4),padding = 'same')(layer5)
#     layer7 = MaxPooling2D((2,2))(layer6)
#     layer8 = Conv2D(256,(4,4),padding = 'same')(layer7)
#     layer9 = MaxPooling2D((2,2))(layer8)
#     layer10 = Deconv2D(512,(4,4),padding = 'same')(layer9)
#     layer11 = UpSampling2D((2,2))(layer10)
#     layer12 = Deconv2D(256,(1,1),padding = 'same')(layer11)
#     layer13 = UpSampling2D((2,2))(layer12)
#     layer14 = Deconv2D(128,(4,4),padding = 'same')(layer13)
#     layer15 = UpSampling2D((2,2))(layer14)
#     layer16 = Deconv2D(64,(4,4),padding = 'same')(layer15)
#     layer17 = UpSampling2D((2,2))(layer16)
#     layer18 = Deconv2D(3,(4,4),padding = 'same')(layer17)
​
  layer = Conv2D(64,(3,3),activation = 'relu', strides=(1,1), padding='same')(input_image)
  for i in range(15):
    layer = Conv2D(64,(3,3),activation = 'relu', strides=(1,1), padding='same')(layer)
    layer = BatchNormalization()(layer)
  layer = Conv2D(3, kernel_size=(3,3), strides=(1,1), padding = 'same')(layer)
  layer = Subtract()([input_image, layer])
  return layer
  
batch_size = 100
epochs = 100
channels = 3
dim = (64,64,channels)
input_image = Input(shape=dim)
​
model = Sequential()
model = Model(input_image, denoiser_model(input_image))
model.compile(loss='mean_squared_error', optimizer = Adam() , metrics=['accuracy'])
model.summary()
​
​
​
weight_directory = '/content/stl10_binary/weights'
​
try:
  os.makedirs(weight_directory, exist_ok=True)
  print("Directory created")
except OSError as exc:
  if exc.errno == errno.EEXIST:
    pass
​
  
filepath = weight_directory + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
​
​
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
​
model_train = model.fit(train_X, train_y, batch_size=batch_size,epochs=epochs,callbacks=callbacks_list,verbose=1,validation_data=(valid_X, valid_y))
​
​
​
​
​
​

[ ]
​
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
​
print("Loading Weights")
model.load_weights('/content/stl10_binary/weights/weights-improvement-04-0.83.hdf5')
print("Loaded Weights")
​
image_path = '/content/image9026.jpg'
​
def gauss_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy
  
pic = cv2.imread(image_path)
reimg = cv2.resize(pic, (64, 64))
reimg = gauss_noise(reimg)
cv2.imwrite('/content/test_noise9026.jpg',reimg)
pred = model.predict(np.expand_dims(reimg,0))
cv2.imwrite('/content/pred9026.jpg',pred[0])

[ ]
from skimage.measure import compare_psnr

ground_truth = cv2.imread('/content/image9026.jpg')
denoised_image = cv2.imread('/content/pred9026.jpg')
psnr_value = compare_psnr(ground_truth,denoised_image)
print(psnr_value)
