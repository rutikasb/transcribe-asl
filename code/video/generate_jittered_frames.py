import os
import argparse
import cv2
import glob
from skimage import io
import numpy as np
from jitter import Jitter

def process_videos(frames_data_path, processed_data_path):
    for d in ['train', 'test']:
        dirs = os.listdir(os.path.join(frames_data_path, d))
        for d2 in dirs:
            if d2.startswith('.'):
                continue
            video_files = os.listdir(os.path.join(frames_data_path, d, d2))
            if video_files:
                for i in range(len(video_files)):
                    video_file = video_files[i]
                    frame_files = sorted(glob.glob(os.path.join(frames_data_path, d, d2, video_file, '*.jpg')))
                    frames = []
                    for frame_file in frame_files:
                        frames.append(io.imread(frame_file))
                    frames = np.array(frames)

                    jiterred_frames = Jitter.get_strobed_videos(frames, 20, None, True, 10, True)
                    if len(jiterred_frames.shape) == 4:
                        jiterred_frames = np.expand_dims(jiterred_frames, axis=0)
                    for j in range(jiterred_frames.shape[0]):
                        result_dir = os.path.join(processed_data_path, d, d2, f'{video_file}_j{j:04d}')
                        os.makedirs(result_dir)
                        for k in range(jiterred_frames.shape[1]):
                            filename = result_dir + f'/frame{k:05d}.jpg'
                            image = jiterred_frames[j][k]
                            io.imsave(filename, image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert single video frames into multiple videos of fixed frame lengths')

    parser.add_argument('--frames-data-path', required=True, help='path to the raw train and test directories')
    parser.add_argument('--processed-data-path', required=True, help='path to the resulting processed data')
    args = vars(parser.parse_args())
    print(args)
    if args.get('frames_data_path') is None or args.get('processed_data_path') is None:
        print('Please provide the path to raw data and path to store processed data')
    else:
        process_videos(args['frames_data_path'], args['processed_data_path'])
