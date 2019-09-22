import argparse
import os

from PIL import Image
from cv2 import resize
from tqdm import tqdm

num_classes = 365
import numpy as np
from tools import get_logger
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load the job offers from different sources to a common ES index")
    parser.add_argument('--input_dir', type=str, default='/data/data2/jmena/Places365-Challenge',
                        help='dir holding the data')
    parser.add_argument('--output_file', type=str, default='places365_test_images.pkl',
                        help='file to dump the generated data')

    logger = get_logger()
    args = parser.parse_args()
    input_dir = args.input_dir
    data_dir = os.path.sep.join([input_dir, 'data_256'])
    output_file = os.path.sep.join([input_dir, args.output_file])
    logger.info('Load pretrained data')
    test_file = os.path.sep.join([input_dir, 'test_data.pkl'])
    with open(test_file, 'rb') as file:
        (test_file_names, test_labels, test_predictions) = pickle.load(file, encoding='latin1')
        test_predictions = np.array(test_predictions)
    test_images = []
    for file_name in tqdm(test_file_names):
        with open(os.path.sep.join([data_dir, file_name]), 'rb') as image_file:
            image = Image.open(image_file)
            image = np.array(image, dtype=np.uint8)
            image = resize(image, (224, 224))
            test_images.append(image)
    test_images = np.array(test_images)

    with open(output_file, 'wb') as file:
        pickle.dump(test_images, file)

    logger.info("Done")
