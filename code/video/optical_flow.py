import os
import argparse
import cv2
import glob
import numpy as np

class OpticalFlow:
    def __init__(self, third_channel=False, f_bound=20.0):
        self.third_channel = third_channel
        self.f_bound = f_bound
        self.prev = np.zeros((1,1))
        self.algorithm = 'tvl1'
        self.oTVL1 = cv2.DualTVL1OpticalFlow_create(scaleStep=0.5, warps=3, epsilon=0.02)

    def first(self, image):
        h, w, _ = image.shape

        self.prev = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        flow = np.zeros((h, w, 2), dtype=np.float32)
        if self.third_channel:
            self.zeros = np.zeros((h, w, 1), dtype=np.float32)
            flow = np.concatenate((arFlow, self.zeros), axis=2)

        return flow

    def next(self, image):
        if self.prev.shape == (1,1): return self.first(image)

        # get image in black&white
        current = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        flow = self.oTVL1.calc(self.prev, current, None)

        # only 2 dims
        flow = flow[:, :, 0:2]

        # truncate to +/-15.0, then rescale to [-1.0, 1.0]
        flow[flow > self.f_bound] = self.f_bound
        flow[flow < -self.f_bound] = -self.f_bound
        flow = flow / self.f_bound

        if self.third_channel:
            # add third empty channel
            flow = np.concatenate((flow, self.zeros), axis=2)

        self.prev = current

        return flow

def files_to_frames(files):
    frames = []
    for f in files:
        frame = cv2.imread(f)
        frames.append(frame)
    return frames

def frames_to_flows(frames):
    of = OpticalFlow()
    flows = []
    for i in range(len(frames)):
        flow = of.next(frames[i])
        flows.append(flow)
    return np.array(flows)

def generate_optical_flows(data_path, output_path):
    for d in ['train', 'test']:
        dirs = glob.glob(os.path.join(data_path, d, '*/*'))
        for d2 in dirs:
            if d2.startswith('.'):
                continue
            print(f'Processing {d2}')
            frame_files = sorted(glob.glob(os.path.join(d2, '*.jpg')))
            if len(frame_files) == 0:
                continue
            frames = files_to_frames(frame_files)
            flows = frames_to_flows(frames)

            output_dir = d2.replace(data_path.replace('/', ''), output_path)
            os.makedirs(output_dir)

            n, h, w, c = flows.shape
            zeros = np.zeros((h, w, 1), dtype=np.float32)
            for i in range(n):
                ar_f_flow = np.concatenate((flows[i], zeros), axis=2)
                ar_n_flow = np.round((ar_f_flow + 1.0) * 127.5).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f'flow{i:05d}.jpg'), ar_n_flow)

if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Convert video frames to optical flow images')

     parser.add_argument('--data-path', required=True, help='path to the train and test directories containing video frames')
     parser.add_argument('--output-path', required=False, default='optical_flow', help='path to processed data')
     args = vars(parser.parse_args())
     print(args)
     if args.get('data_path') is None:
         print('provide data path to frames')
     else:
         generate_optical_flows(args['data_path'], args['output_path'])
