import glob
import os
import sys
import cv2
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder

class FramesSeqGenerator(keras.utils.Sequence):
    def __init__(self, data_path, batch_size, num_frames_per_video,
            height, width, colour_channels, shuffle=True):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_frames = num_frames_per_video
        # 3 for raw frame or image and 2 for optical flow image
        self.colour_channels = colour_channels
        self.input_shape = (num_frames_per_video, height, width, colour_channels)

        self.videos_list = pd.DataFrame(sorted(glob.glob(data_path + '/*/*')), columns=["frames_dir"])
        self.num_samples = len(self.videos_list)

        labels = self.videos_list.frames_dir.apply(lambda s: s.split("/")[-2])
        self.videos_list.loc[:, "label"] = labels

        # extract unique classes from all detected labels
        self.classes = sorted(list(self.videos_list.label.unique()))
        print(f'Classes = {self.classes}')
        self.num_classes = len(self.classes)

        # encode labels
        le = LabelEncoder()
        le.fit(self.classes)
        self.videos_list.loc[:, "label_num"] = le.transform(self.videos_list.label)

        self.on_epoch_end()
        return

    def __len__(self):
        return int(np.ceil(self.num_samples/self.num_frames))

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        return

    def __get_item__(self, step):
        indexes = self.indexes[step*self.batch_size:(step+1)*self.batch_size]
        batch = self.videos_list.loc[indexes, :]
        batch_x = np.empty((batch_size, ) + self.input_shape, dtype=float)
        batch_y = np.empty((batch_size), dtype=int)

        for i in range(batch_size):
            batch_x[i,], batch_y[i] = self.generate(batch.iloc[i,:])

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.num_classes)

    def read_frame_files(self, frames_dir):
        print(frames_dir)
        files = sorted( glob.glob(frames_dir + "/*.jpg"))
        frames = []
        for f in files:
            frame = cv2.imread(f)
            frames.append(frame)
        return np.array(frames)

    def normalize_frames(self, frames, target_num_of_frames):
        samples, _, _, _ = frames.shape
        if samples == target_num_of_frames:
            return frames

        portion = samples/target_num_of_frames
        index = [int(portion * i) for i in range(target_num_of_frames)]
        result = [frames[i, :, :, :] for i in index]
        return np.array(result)

    def generate(self, video):
        frames_list = self.read_frame_files(video.frames_dir)
        # print(frames_list.shape)
        frames_list = frames_list[:, :, :, 0:self.colour_channels]
        frames = self.normalize_frames(frames_list, self.num_frames)

        return frames, video.label


class FeaturesSeqGenerator(keras.utils.Sequence):
    def __init__(self, data_path, batch_size, input_shape, shuffle=True):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle

        self.samples = pd.DataFrame(sorted(glob.glob(data_path + "/*/*.npy")), columns=["path"])
        self.num_samples = len(self.samples)
        # test shape of a sample
        # x = np.load(self.samples.path[0])
        # if x.shape != input_shape: raise ValueError(f'Incorrect feature shape: {x.shape} against {input_shape}')

        # extract (text) labels from path
        labels =  self.samples.path.apply(lambda s: s.split("/")[-2])
        self.samples.loc[:, "label"] = labels

        self.classes = sorted(list(self.samples.label.unique()))
        self.num_classes = len(self.classes)

        # encode labels
        le = LabelEncoder()
        le.fit(self.classes)
        self.samples.loc[:, "label"] = le.transform(self.samples.label)

        self.on_epoch_end()
        return

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        return

    def __getitem__(self, step):
        indexes = self.indexes[step*self.batch_size:(step+1)*self.batch_size]

        batch = self.samples.loc[indexes, :]
        batch_size = len(batch)

        batch_x = np.empty((batch_size, ) + self.input_shape, dtype=float)
        batch_y = np.empty((batch_size), dtype=int)

        for i in range(batch_size):
            batch_x[i,], batch_y[i] = self.generate(batch.iloc[i,:])

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.num_classes)

    def generate(self, sample):
        x = np.load(sample.path)
        return x, sample.label
