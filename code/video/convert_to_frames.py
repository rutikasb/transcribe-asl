import os
import argparse
import cv2

def process_videos(raw_data_path, processed_data_path):
    for d in ['train', 'test']:
        dirs = os.listdir(os.path.join(raw_data_path, d))
        for d2 in dirs:
            video_files = os.listdir(os.path.join(raw_data_path, d, d2))
            if video_files:
                for i in range(len(video_files)):
                    result_dir = os.path.join(processed_data_path, d, d2, f'video_{i}')
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)
                    video_file = os.path.join(raw_data_path, d, d2, video_files[i])
                    cap = cv2.VideoCapture(video_file)
                    success, image = cap.read()
                    count = 0
                    while success:
                        image = cv2.resize(image, (480, 480), interpolation = cv2.INTER_AREA)
                        filename = os.path.join(result_dir, f'frame{count:05d}.jpg')
                        print(filename)
                        cv2.imwrite(filename, image)
                        success, image = cap.read()
                        count += 1
                    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert videos in a given path to frames')

    parser.add_argument('--raw-data-path', required=True, help='path to the raw train and test directories')
    parser.add_argument('--processed-data-path', required=True, help='path to the resulting processed data')
    args = vars(parser.parse_args())
    print(args)
    if args.get('raw_data_path') is None or args.get('processed_data_path') is None:
        print('Please provide the path to raw data and path to store processed data')
    else:
        process_videos(args['raw_data_path'], args['processed_data_path'])
