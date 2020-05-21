import math
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import glob
import os
from numpy import exp
import matplotlib.pyplot as plt

def preprocess_image(im):
    width, height = im.size
    if (height > width):
        im = im.transpose(Image.ROTATE_90)

    size = 64, 64
    im = im.convert('L')
    im.thumbnail(size, Image.ANTIALIAS)
    data = np.asarray(im)
    pixel_array = data.flatten()

    return pixel_array


def normalize_image(im):
    pixels = np.interp(im, (0, 255), (-1, 1))
    return pixels

def normalize_images(X_set):
    n = np.shape(X_set)[0]
    res = np.zeros(shape=(n,4096))
    for i in range(n):
        res[i] = normalize_image(X_set[i,:])

    return res


def load_data(folder):

    path, dirs, files = next(os.walk('/home/tunde/Linux/Documents/Harmadev-II/ai/delta/second_v/' + folder))
    file_count = len(files)
    images = np.zeros(shape=(file_count,4096))
    i = 0

    for filename in glob.glob('/home/tunde/Linux/Documents/Harmadev-II/ai/delta/second_v/' + folder + '/*.jpg'):

        im = Image.open(filename)
        im = preprocess_image(im)
        padding_nr = 4096 - len(im)
        im = np.pad(im,(0,padding_nr),'constant',constant_values=0)
        images[i] = im
        i = i + 1

    return images


def split_data(folder1, folder2, test_size):
    images1 = load_data(folder1)
    number_of_images1 = len(images1)

    images2 = load_data(folder2)
    number_of_images2 = len(images2)

    images = np.concatenate((images1,images2),axis=0)
    number_of_images = len(images)

    y1 = np.zeros(number_of_images1)
    y2 = np.ones(number_of_images2)
    y = np.concatenate((y1,y2),axis=0)
    y = y.reshape(number_of_images,1)

    X_train, X_test, y_train, y_test = train_test_split(images,y, test_size=test_size,random_state=42)
    return X_train, X_test, y_train, y_test

def f(x):
    x = np.interp(x, (x.min(), x.max()), (-10, 10))
    activation = 1.0 / (1 + exp(-x))
    return activation

def gradf(x):
    #dif = np.subtract(1,f(x))
    #return np.multiply(f(x),dif)
    return f(x)*(1-f(x))

def offlineLearning(x, d, lr):
    n = np.shape(x)[1]
    w = np.random.rand(n,1)
    iter = 0
    E = []

    while True:
        v = x.dot(w)
        y = f(v)
        e = y - d
        grad = gradf(v)

        dif = np.multiply(e, grad)
        x_transpose = np.transpose(x)
        g = x_transpose.dot(dif)

        w = w - (lr * g)

        square_loss_value = np.sum(np.power(e,2))
        E.append(square_loss_value)

        if iter > 10:
            if iter > 10000:
                print(iter)
                return w,E
            elif E[-10] < E[-1] or E[-10] - E[-1] < 0.001:
                return w, E
            else:
                pass
        else:
            pass

        iter = iter + 1

def predict_test_data(X_test,w):
    act = X_test.dot(w)
    act = f(act)
    act = np.around(act)
    return act

def check_prediction(predicted,y_test):
    return predicted == y_test

def calc_score(res):
    input_nr = len(res)
    true_pozitive = np.count_nonzero(res)
    return float(true_pozitive)/input_nr

def plot_test_images(X_test,predicted):
    test_nr = np.shape(X_test)[0]
    test_nr /= 2
    grid_size = int(round(math.sqrt(test_nr)-1))
    fig = plt.figure(figsize=(grid_size,grid_size))
    index = 0

    for i in range(grid_size):
        for j in range(grid_size):
            array_im = X_test[index,:]
            mat = np.reshape(array_im,(64,64))
            im = Image.fromarray(np.uint8(mat*255),'L')
            plt.subplot(grid_size,grid_size,index+1)
            if (predicted[index] == 0):
                label = "revolver"
            else:
                label = "mandolin"
            plt.title(label)
            plt.xticks([]),plt.yticks([])
            plt.tight_layout()
            plt.imshow(im, cmap='gray',vmin=0, vmax=255)
            index += 1
    plt.show()

def plot_confusion_matrix(y_test, predicted):
    labels = ['revolver','mandolin']
    cm = confusion_matrix(y_test,predicted)
    print(cm)
    ax = plt.subplot()
    sn.heatmap(cm,annot=True, ax=ax)
    ax.set_xlabel('Predicted');
    ax.set_ylabel('True');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(labels);
    ax.yaxis.set_ticklabels(labels);

    plt.show()

# ----------------------------------------------------------


test_size = 0.3
learning_rate = 0.01


X_train, X_test, y_train, y_test = split_data('revolver', 'mandolin', test_size)
#X_train, X_test, y_train, y_test = split_data('airplanes', 'motorbikes', test_size)

X_test_original = X_test
X_train = normalize_images(X_train)
X_test = normalize_images(X_test)


w,E = offlineLearning(X_train,y_train,learning_rate)

print(E)

predicted = predict_test_data(X_test,w)
res = check_prediction(predicted,y_test)

score = calc_score(res)
print("Score")
print(score)

plot_test_images(X_test_original,predicted)
plot_confusion_matrix(y_test, predicted)





