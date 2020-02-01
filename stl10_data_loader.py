from tensorflow.python.keras.utils import to_categorical
import numpy as np
import sys
import os
import urllib.request as urllib
import tarfile

# the dimensions of each image in the STL-10 dataset (96x96x3).
HEIGHT, WIDTH, DEPTH = 96, 96, 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
# DATA_DIR = '/data/data2/jmena/STL-10'
DATA_DIR = 'stl10_data'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
TRAIN_DATA_PATH = DATA_DIR + '/stl10_binary/train_X.bin'

# path to the binary test file with image data
TEST_DATA_PATH = DATA_DIR + '/stl10_binary/test_X.bin'

# path to the binary train file with labels
TRAIN_LABELS_PATH = DATA_DIR + '/stl10_binary/train_y.bin'

# path to the binary test file with labels
TEST_LABELS_PATH = DATA_DIR + '/stl10_binary/test_y.bin'

# path to class names file
CLASS_NAMES_PATH = DATA_DIR + '/stl10_binary/class_names.txt'

# path to the binary unlabeled file with image data
UNLABELED_DATA_PATH = DATA_DIR + '/stl10_binary/unlabeled_X.bin'


class STL10Loader(object):

    def __init__(self, num_classes):
        assert num_classes in [2, 10]
        self.num_classes = num_classes

    def read_labels(self, path_to_labels):
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    def read_all_images(self, path_to_data):
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

            images = np.reshape(everything, (-1, DEPTH, WIDTH, HEIGHT))

            # Now transpose the images into a standard image format
            # readable by, for example, matplotlib.imshow
            # You might want to comment this line or reverse the shuffle
            # if you will use a learning algorithm like CNN, since they like
            # their channels separated.
            images = np.transpose(images, (0, 3, 2, 1))
            return images

    def download_and_extract(self):
        # if the dataset already exists locally, no need to download it again.
        if all((
                os.path.exists(TRAIN_DATA_PATH),
                os.path.exists(TRAIN_LABELS_PATH),
                os.path.exists(TEST_DATA_PATH),
                os.path.exists(TEST_LABELS_PATH),
        )):
            return

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

    def load_dataset(self):
        # download the extract the dataset.
        self.download_and_extract()

        # load the train and test data and labels.
        x_train = self.read_all_images(TRAIN_DATA_PATH)
        y_train = self.read_labels(TRAIN_LABELS_PATH)
        x_test = self.read_all_images(TEST_DATA_PATH)
        y_test = self.read_labels(TEST_LABELS_PATH)

        # convert all images to floats in the range [0, 1]
        x_train = x_train.astype('float32')
        # x_train = (x_train + 53 - 127.5) / 127.5
        x_train = (x_train - 10) / 255.0
        x_test = x_test.astype('float32')
        x_test = (x_test - 10) / 255.0
        # x_test = (x_test +53 - 127.5) / 127.5

        # convert the labels to be zero based.
        y_train -= 1
        y_test -= 1

        if self.num_classes == 2:
            # filter only the first 2 cifar10 classes
            x_train = x_train[np.isin(y_train, [0, 2])]
            y_train = y_train[np.isin(y_train, [0, 2])]
            x_test = x_test[np.isin(y_test, [0, 2])]
            y_test = y_test[np.isin(y_test, [0, 2])]

            # create cifar10 equivalent labels
            y_train_cifar10 = np.copy(y_train)
            y_test_cifar10 = np.copy(y_test)
            # we fix classes bird(1) and car(2) which are swapped in the two datasets
            y_train_cifar10[y_train_cifar10 == 2] = 1
            y_test_cifar10[y_test_cifar10 == 2] = 1
        else:
            # create cifar10 equivalent labels
            y_train_cifar10 = np.copy(y_train)
            y_test_cifar10 = np.copy(y_test)
            # we fix classes bird(1) and car(2) which are swapped in the two datasets
            y_train_cifar10[y_train_cifar10 == 1] = 10
            y_train_cifar10[y_train_cifar10 == 2] = 1
            y_train_cifar10[y_train_cifar10 == 10] = 2
            y_test_cifar10[y_test_cifar10 == 1] = 10
            y_test_cifar10[y_test_cifar10 == 2] = 1
            y_test_cifar10[y_test_cifar10 == 10] = 2
            # we fix classes horse(6) and monkey(7) which are swapped in the two datasets
            # in this case, the resulting class 6, monkey, differs from CIFAR, where it
            # corresponds to frogs
            y_train_cifar10[y_train_cifar10 == 6] = 10
            y_train_cifar10[y_train_cifar10 == 7] = 6
            y_train_cifar10[y_train_cifar10 == 10] = 7
            y_test_cifar10[y_test_cifar10 == 6] = 10
            y_test_cifar10[y_test_cifar10 == 7] = 6
            y_test_cifar10[y_test_cifar10 == 10] = 7

        # convert labels to hot-one vectors.
        y_train_cat = to_categorical(y_train_cifar10, self.num_classes)
        y_test_cat = to_categorical(y_test_cifar10, self.num_classes)

        return (x_train, y_train, y_train_cifar10, y_train_cat), (x_test, y_test, y_test_cifar10, y_test_cat)

    def load_raw_dataset(self):
        # download the extract the dataset.
        self.download_and_extract()

        # load the train and test data and labels.
        x_train = self.read_all_images(TRAIN_DATA_PATH)
        y_train = self.read_labels(TRAIN_LABELS_PATH)
        x_test = self.read_all_images(TEST_DATA_PATH)
        y_test = self.read_labels(TEST_LABELS_PATH)

        # convert all images to floats in the range [0, 1]
        x_train = x_train.astype('float32')
        x_train = (x_train - 10) / 255.0
        x_test = x_test.astype('float32')
        x_test = (x_test - 10) / 255.0

        # convert the labels to be zero based.
        y_train -= 1
        y_test -= 1

        # convert labels to hot-one vectors.
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_test_cat = to_categorical(y_test, self.num_classes)

        return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat)

    def load_raw_ood_dataset(self, num_training=500, num_test=500):
        # download the extract the dataset.
        self.download_and_extract()

        # load the train and test data and labels.
        unlabeled = self.read_all_images(UNLABELED_DATA_PATH)

        # convert all images to floats in the range [0, 1]
        unlabeled = unlabeled.astype('float32')
        unlabeled = (unlabeled - 10) / 255.0
        np.random.shuffle(unlabeled)

        return unlabeled[:num_training], unlabeled[num_training:num_test]
