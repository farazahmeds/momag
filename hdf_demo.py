from neurovc.util.IO_util import Debayerer, VideoWriter
from neurovc.momag import OnlineLandmarkMagnifier
from neurovc.momag import ThreshCompressor
import h5py
import cv2
import neurovc as nvc
import numpy as np


from neurovc.momag import ThreshCompressor

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./configs", config_name='config', version_base='1.3')
def main(config: DictConfig):
    comp = ThreshCompressor(alpha=config.alpha, threshold=config.threshold)

    debayerer = Debayerer()
    motion_magnifier = None

    writer = VideoWriter(f"{config.output_file}", framerate=config.output_frame_rate)

    with h5py.File(config.input_file, "r") as f:
        for i in range(config.trigger_frame - config.crop_frame_range, config.trigger_frame + config.crop_frame_range, config.step_frame):
            frame = f["Frames"][i]
            frame = cv2.resize(debayerer(frame), None, fx=0.5, fy=0.5)

            ref = cv2.resize(debayerer(f["Frames"][0]), None, fx=0.5, fy=0.5)

            if motion_magnifier is None:
                motion_magnifier = OnlineLandmarkMagnifier(landmarks=nvc.LM_MOUTH + nvc.LM_EYE_LEFT + nvc.LM_EYE_RIGHT,
                                                           reference=ref, alpha=config.alpha, attenuation_function=comp)

            magnified, _ = motion_magnifier(frame)
            frame = np.concatenate((frame, magnified), axis=1)
            writer(frame)


if __name__ == "__main__":
    main()