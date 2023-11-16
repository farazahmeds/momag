__author__ = 'Philipp Flotho'


from neurovc.util.IO_util import Debayerer, VideoWriter
from neurovc.momag import OnlineLandmarkMagnifier
import h5py
import cv2
import neurovc as nvc
import numpy as np
import sys


from neurovc.momag import ThreshCompressor


def run_momag(path_to_hdf_file, alpha, threshold, frame_rate, file_name, frames_list):

    for index, segmented_frames in enumerate(frames_list):
        first_frame, second_frame = segmented_frames

        comp = ThreshCompressor(alpha=alpha, threshold=threshold)

        input_file = str(path_to_hdf_file)

        debayerer = Debayerer()
        motion_magnifier = None

        writer = VideoWriter(f"{str(file_name)}_seg_{index}.mp4", framerate=frame_rate)

        with h5py.File(input_file, "r") as f:
            for i in range(first_frame, second_frame, 3):
                frame = f["Frames"][i]
                frame = cv2.resize(debayerer(frame), None, fx=0.5, fy=0.5)

                ref = cv2.resize(debayerer(f["Frames"][0]), None, fx=0.5, fy=0.5)

                if motion_magnifier is None:
                    motion_magnifier = OnlineLandmarkMagnifier(landmarks=nvc.LM_MOUTH + nvc.LM_EYE_LEFT + nvc.LM_EYE_RIGHT,
                                                               reference=ref, alpha=alpha, attenuation_function=comp)

                magnified, _ = motion_magnifier(frame)
                frame = np.concatenate((frame, magnified), axis=1)
                writer(frame)

    sys.exit("Error message")


