from neurovc.util.IO_util import Debayerer, VideoWriter
from neurovc.momag import OnlineLandmarkMagnifier
import h5py
import cv2
import neurovc as nvc
import numpy as np
import os
from paired_frames_from_trigger import PairedFramesBetweenFrames
from neurovc.momag import ThreshCompressor
from tqdm import tqdm
import glob
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
import difflib


@hydra.main(config_path="./configs/dils", config_name='config')
def main(config: DictConfig):

    # trigger files parse

    trigger_string_ = ''.join(['P',str((config.study.subject_id.split('_')[1])),'B', f'{config.study.blocks}'])
    trigger_files = [os.path.basename(x) for x in glob.glob(f"{config.work_dir}/Data/trigger_files/*.txt")]
    sep_ = ['_'.join(x.split('_')[0:1]) for x in trigger_files]
    time_stamp_id = difflib.get_close_matches(f'{trigger_string_}', sep_)[0]

    comp = ThreshCompressor(alpha=config.motion_magnification.alpha, threshold=config.motion_magnification.threshold)

    # faces files parse

    faces_files = glob.glob(f"{config.work_dir}/Data/faces/*.txt")

    def similarity_score(s1, s2):
        """Calculate the number of matching characters in the same position."""
        return sum(1 for a, b in zip(s1, s2) if a == b)

    # Remove the first character from each filename and compute similarity
    scores = []
    for path in faces_files:
        filename = os.path.basename(path)[:-4]  # Remove extension and get the filename
        modified_filename = filename[1:]  # Ignore the first character
        score = similarity_score(modified_filename, time_stamp_id)
        scores.append((path, score))

    # Find the path with the highest score
    face_file_path = max(scores, key=lambda x: x[1])[0]

    print (face_file_path)

    try:
        with open(f'{face_file_path}', 'r') as file:
            # Read the file content
            content = file.read()
            # Split the content by comma and convert each item to an integer
            integers = [int(item) for item in content.split(',')]

        # Finding the index of the first '0' integer
        reference_frame_neutral = integers.index(1)
        print(f"The index of the first '1' is: {reference_frame_neutral}")

    except FileNotFoundError:
        print(f"File not found: {face_file_path}")
    except ValueError as e:
        # This will catch both conversion errors and not finding '0'
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


    if config.study.visual_trigger:
        time_stamps = [rf'{config.work_dir}/Data/trigger_files/{time_stamp_id}_vis.txt']
    else:
        time_stamps = [rf'{config.work_dir}/Data/trigger_files/{time_stamp_id}_aud.txt',
                       rf'{config.work_dir}/Data/trigger_files/{time_stamp_id}_vis.txt']

    hdf_files = glob.glob(rf'{config.input.path_to_hdf_video}\{config.study.subject_id}\*\*.h5')

    os.makedirs(f'{config.output.output_path}\{config.study.subject_id}', exist_ok=True)

    config_dict = OmegaConf.to_container(config, resolve=True)

    with open(fr"{config.output.output_path}\{config.study.subject_id}\config.yaml", 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)

    for time_stamp_file in time_stamps:

        paired = PairedFramesBetweenFrames(path_to_hdf=hdf_files[config.study.blocks-1],
                                           path_to_time_stamps_file=f'{time_stamp_file}')

        paired_frames = paired.get_paired_frames()
        debayerer = Debayerer()
        motion_magnifier = None

        def prefix(file_path=time_stamp_file):

            file_name = os.path.basename(file_path)
            part = file_name.split('_')[1]
            return part.split('.')[0]

        for i, pair in enumerate(tqdm(paired_frames, colour='GREEN')):

            writer = VideoWriter(fr"{config.output.output_path}\{config.study.subject_id}\{config.study.subject_id}_{prefix()}_{i}.mp4", framerate=config.output.video_frame_rate)

            with h5py.File(hdf_files[config.study.blocks-1], "r") as f:
                for i in range(pair[0], pair[2], 1):
                    frame = f["Frames"][i]
                    frame = cv2.resize(debayerer(frame), None, fx=config.motion_magnification.frame_resize, fy=config.motion_magnification.frame_resize)

                    ref = cv2.resize(debayerer(f["Frames"][int(reference_frame_neutral)]), None, fx=config.motion_magnification.frame_resize, fy=config.motion_magnification.frame_resize)

                    if motion_magnifier is None:
                        motion_magnifier = OnlineLandmarkMagnifier(
                            landmarks=nvc.LM_MOUTH + nvc.LM_EYE_LEFT + nvc.LM_EYE_RIGHT,
                            reference=ref, alpha=config.motion_magnification.alpha, attenuation_function=comp)

                    magnified, _ = motion_magnifier(frame)
                    frame = np.concatenate((frame, magnified), axis=1)

                    for write_redundant_frames in range(config.output.video_frame_rate):
                        writer(frame)


if __name__ == "__main__":
    main()
