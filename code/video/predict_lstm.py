import os
import json
import cv2
import argparse
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model, Model
from optical_flow import frames_to_flows

def get_frames(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    success, image = cap.read()
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (299, 299), interpolation = cv2.INTER_AREA)
        height, width, depth = image.shape
        sX = int(width/2 - 299/2)
        sY = int(height/2 - 299/2)
        image = image[sY:sY+299, sX:sX+299, :]
        frames.append(image)
        success, image = cap.read()
    cap.release()
    return np.array(frames)

def normalize_frames(frames, target_num_of_frames):
        samples, _, _, _ = frames.shape
        if samples == target_num_of_frames:
            return frames

        portion = samples/target_num_of_frames
        index = [int(portion * i) for i in range(target_num_of_frames)]
        result = [frames[i, :, :, :] for i in index]
        return np.array(result)

def predict(filename, model_path, labels_path):
    frames = get_frames(filename)
    frames= frames[:, :, :, 0:3]
    frames = normalize_frames(frames, 20)
    # flows = frames_to_flows(frames)

    # x = np.array(flows)
    x = np.array(frames)
    print(x.shape)
    # model_inception = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    model_inception = InceptionV3(weights='imagenet', include_top=True)
    base_model = Model(inputs=model_inception.input, outputs=model_inception.get_layer('avg_pool').output)
    model = load_model(model_path)

    x_features = base_model.predict(x)
    predictions = model.predict(np.expand_dims(x_features, axis=0))
    print(f'Predictions = {predictions}')

    labels = sorted(os.listdir('frames_data_small/train'))
    rev_labels = dict()
    for i in range(len(labels)):
        rev_labels[i] = labels[i]

    top_5 = predictions[0].argsort()[-5:][::-1]
    results = []
    for index in top_5:
        results.append(rev_labels[index])

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given a video snippet, predict the class')

    parser.add_argument('--video-file',  required=True, help='path to the video file')
    parser.add_argument('--model', required=True, help='path to saved model')
    parser.add_argument('--labels', required=False, default='labels.txt', help='path to stored labels file')
    args = vars(parser.parse_args())
    if args.get('video_file') is None or args.get('model') is None:
        print('Please provide path to both video file and model on which to predict')
        exit()

    results = predict(args['video_file'], args['model'], args['labels'])
    print(results)
