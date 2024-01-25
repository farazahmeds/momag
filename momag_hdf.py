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
    def load_trigger_and_faces_logs(subject_id, study_block, working_dir):
        # trigger files parse
        trigger_string_ = ''.join(['P',str((subject_id)),'B', f'{study_block}'])
        trigger_files = [os.path.basename(x) for x in glob.glob(f"{working_dir}/data/trigger_files/*.txt")]
        sep_ = ['_'.join(x.split('_')[0:1]) for x in trigger_files]
        time_stamp_id = difflib.get_close_matches(f'{trigger_string_}', sep_)[0]

        # faces files parse

        faces_files = glob.glob(f"{working_dir}/data/faces/*.txt")

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

        return face_file_path, time_stamp_id

    def reference_frame(reference_emotion, face_file_log_path):
        # Reference frame grab
        try:
            with open(f'{face_file_log_path}', 'r') as file:
                # Read the file content
                content = file.read()
                # Split the content by comma and convert each item to an integer
                integers = [int(item) for item in content.split(',')]

            # Finding the index of the first '0' integer
            if reference_emotion == 'neutral':
                frame_nr = 1
                reference_frame_neutral = integers.index(frame_nr)
                return reference_frame_neutral

        except FileNotFoundError:
            print(f"File not found: {face_file_path}")
        except ValueError as e:
            # modified laura's face files
            with open(face_file_log_path, 'r') as file:
                for index, line in enumerate(file):
                    if line.startswith('N') and reference_emotion == 'neutral':
                        return index

        except Exception as e:
            print(f"An error occurred: {e}")

    def motion_magnification(path_to_recording_root_folder,
                             path_to_working_dir,
                             time_stamp_id,
                             include_only_visual_trigger,
                             subject_id,
                             blocks,
                             output_path,
                             reference_frame,
                             alpha,
                             threshold,
                             frame_resize,
                             output_video_frame_rate
                             ):

        if include_only_visual_trigger:
            time_stamps = [rf'{path_to_working_dir}/data/trigger_files/{time_stamp_id}_vis.txt']
        else:
            time_stamps = [rf'{path_to_working_dir}/data/trigger_files/{time_stamp_id}_aud.txt',
                           rf'{path_to_working_dir}/data/trigger_files/{time_stamp_id}_vis.txt']

        hdf_files = glob.glob(rf'{path_to_recording_root_folder}\{subject_id}*\*.h5')

        comp = ThreshCompressor(alpha=alpha, threshold=threshold)

        config_dict = OmegaConf.to_container(config, resolve=True)

        with open(fr"{output_path}\{subject_id}\config.yaml", 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)

        for time_stamp_file in time_stamps:

            paired = PairedFramesBetweenFrames(path_to_hdf=hdf_files[blocks-1],
                                               path_to_time_stamps_file=f'{time_stamp_file}')

            paired_frames = paired.get_paired_frames()

            debayerer = Debayerer()
            motion_magnifier = None

            def prefix(file_path=time_stamp_file):

                file_name = os.path.basename(file_path)
                part = file_name.split('_')[1]
                return part.split('.')[0]

            for i, pair in enumerate(tqdm(paired_frames, colour='GREEN')):

                writer = VideoWriter(fr"{output_path}\{subject_id}\{subject_id}_{prefix()}_{i}.mp4",
                                     framerate=output_video_frame_rate)

                with h5py.File(hdf_files[blocks-1], "r") as f:
                    for i in range(pair[0], pair[2], 1):
                        frame = f["Frames"][i]
                        frame = cv2.resize(debayerer(frame), None, fx=frame_resize, fy=frame_resize)

                        ref = cv2.resize(debayerer(f["Frames"][int(reference_frame)]),
                                         dsize=None,
                                         fx=frame_resize,
                                         fy=frame_resize)

                        if motion_magnifier is None:
                            motion_magnifier = OnlineLandmarkMagnifier(
                                landmarks=nvc.LM_MOUTH + nvc.LM_EYE_LEFT + nvc.LM_EYE_RIGHT,
                                reference=ref,
                                alpha=alpha,
                                attenuation_function=comp)

                        magnified, _ = motion_magnifier(frame)
                        frame = np.concatenate((frame, magnified), axis=1)

                        for write_redundant_frames in range(output_video_frame_rate):
                            writer(frame)

    os.makedirs(f'{config.output.output_path}\{config.study.subject_id}', exist_ok=True)

    face_file_path, time_stamp_id = load_trigger_and_faces_logs(subject_id=config.study.subject_id,
                                                                study_block=config.study.blocks,
                                                                working_dir=config.work_dir)

    ref_frame = reference_frame(reference_emotion='neutral', face_file_log_path=face_file_path)

    motion_magnification(path_to_recording_root_folder=config.input.path_to_recordings,
                         path_to_working_dir=config.work_dir,
                         time_stamp_id=time_stamp_id,
                         subject_id=config.study.subject_id,
                         blocks=config.study.blocks,
                         reference_frame=ref_frame,
                         include_only_visual_trigger=config.study.visual_trigger,
                         output_path=config.output.output_path,
                         alpha=config.motion_magnification.alpha,
                         threshold=config.motion_magnification.threshold,
                         frame_resize=config.motion_magnification.frame_resize,
                         output_video_frame_rate=config.output.video_frame_rate)


if __name__ == "__main__":
    main()
