import argparse

import os
from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.applications.mobilenet_v2 import decode_predictions

import numpy as np
from stl10_data_loader import STL10Loader
from tools import get_logger
import skimage
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict STL-10 using ImageNet model")
    parser.add_argument('--save_dir', type=str, default='./',
                        help='dir to save the trained model')
    parser.add_argument('--label_mapping_file', type=str, default='imagenet_stl10_mapping.pkl',
                        help='label mapping file')

    args = parser.parse_args()
    save_dir = args.save_dir
    logger = get_logger()
    new_shape = (224, 224, 3)
    num_classes = 10
    logger.error("Loading STL-10")
    stl10_data_loader = STL10Loader(num_classes)
    (stl10_x_train, stl10_y_train, _), (stl10_x_test, stl10_y_test, _) = stl10_data_loader.load_raw_dataset()
    logger.error("Resize training images")
    stl10_x_train_resized = np.array(
        [skimage.transform.resize(image, new_shape, anti_aliasing=True) for image in stl10_x_train])
    # logger.error("Store train data")
    # with open(os.path.sep.join([args.save_dir, 'stl10_x_train_resized.pkl']), 'wb') as file:
    #     pickle.dump((stl10_x_train_resized, stl10_y_train), file)
    logger.error("Resize test images")
    stl10_x_test_resized = np.array(
        [skimage.transform.resize(image, new_shape, anti_aliasing=True) for image in stl10_x_test])
    # logger.error("Store test data")
    # with open(os.path.sep.join([args.save_dir, 'stl10_x_test_resized.pkl']), 'wb') as file:
    #     pickle.dump((stl10_x_test_resized, stl10_y_test), file)
    logger.error("Load model")
    model = mobilenet_v2.MobileNetV2(weights='imagenet')
    logger.error("Predict training")
    train_preds = model.predict(stl10_x_train_resized)
    logger.error("Store train predictions")
    with open(os.path.sep.join([args.save_dir, 'train_preds.pkl']), 'wb') as file:
        pickle.dump(train_preds, file)
    logger.error("Predict test")
    test_preds = model.predict(stl10_x_test_resized)
    logger.error("Store test predictions")
    with open(os.path.sep.join([args.save_dir, 'test_preds.pkl']), 'wb') as file:
        pickle.dump(test_preds, file)

    with open(args.label_mapping_file, 'rb') as file:
        imagenet_stl10_mapping = pickle.load(file)

    pred_y = [imagenet_stl10_mapping[label[0][1]] for label in decode_predictions(test_preds, top=1)]
    acc = np.sum(pred_y == stl10_y_test) / test_preds.shape[0]
    logger.error("Test accuracy {}".format(acc))
    logger.error("Done")
