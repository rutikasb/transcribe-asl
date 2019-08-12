import os
import argparse
import cv2
from skimage import io
import numpy as np

def process_videos(raw_data_path, processed_data_path):
    for d in ['train', 'test']:
        dirs = os.listdir(os.path.join(raw_data_path, d))
        for d2 in dirs:
            if d2.startswith('.'):
                continue
            video_files = os.listdir(os.path.join(raw_data_path, d, d2))
            if video_files:
                for i in range(len(video_files)):
                    video_file = os.path.join(raw_data_path, d, d2, video_files[i])
                    cap = cv2.VideoCapture(video_file)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if(total_frames < 20):
                        print("Skipping {0}, less than 20 frame".format(video_file))
                        continue

                    result_dir = os.path.join(processed_data_path, d, d2, f'video_{i}')
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    success, image = cap.read()
                    prvs = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    hsv = np.zeros_like(image)
                    hsv[...,1] = 255
                    count = 0
                    while success:
                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        success, frame2 = cap.read()
                        if not success:
                            break
                        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 9, 3, 5, 1.2, 0)
                        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                        hsv[...,0] = ang*180/np.pi/2
                        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                        image = cv2.resize(rgb, (299, 299), interpolation = cv2.INTER_AREA)
                        filename = os.path.join(result_dir, f'frame{count:05d}.jpg')
                        print(filename)
                        cv2.imwrite(filename, image)
                        prvs = next
                        # io.imsave(filename, image)
                        # success, image = cap.read()
                        count += 1
                    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert videos in a given path to frames')

    parser.add_argument('--raw-data-path', required=True, help='path to the raw train and test directories')
    parser.add_argument('--processed-data-path', required=True, help='path to the resulting processed data')
    # parser.add_argument('--rescale-by-centering', required=False, default=False, help='whether to center and resize the frames to 299x299')
    args = vars(parser.parse_args())
    print(args)
    if args.get('raw_data_path') is None or args.get('processed_data_path') is None:
        print('Please provide the path to raw data and path to store processed data')
    else:
        process_videos(args['raw_data_path'], args['processed_data_path'])
