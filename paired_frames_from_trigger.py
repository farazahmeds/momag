import h5py
from moviepy.editor import clips_array, VideoFileClip
import glob

class PairedFramesBetweenFrames(object):
    def __init__(self, path_to_hdf, path_to_time_stamps_file):
        self.path_to_hdf = path_to_hdf
        self.path_to_time_stamps_file = path_to_time_stamps_file

    def get_paired_frames(self):
        def elapsed_time(reference_timestamp_ms, current_timestamp_ms):
            """
            Calculate the elapsed time in seconds and milliseconds from a reference timestamp.

            :param reference_timestamp_ms: The reference timestamp in milliseconds.
            :param current_timestamp_ms: The current timestamp in milliseconds.
            :return: A string representing the elapsed time in the format "seconds.milliseconds".
            """
            elapsed_ms = current_timestamp_ms - reference_timestamp_ms
            elapsed_seconds = str(elapsed_ms // 1000).strip('0')
            elapsed_milliseconds = elapsed_ms % 1000
            return f"{elapsed_seconds}{str(elapsed_milliseconds).zfill(2).strip('0')}"

        def read_numbers_from_file(file_path):
            numbers = []  # List to hold the numbers
            with open(file_path, 'r') as file:  # Open the file in read mode
                for line in file:  # Iterate over each line in the file
                    try:
                        number = float(line.strip())  # Convert line to a float
                        numbers.append(number)  # Add the number to the list
                    except ValueError:
                        print(f"Skipping invalid line: {line.strip()}")

            return numbers[:35]

        def read_hdf_file(path_to_hdf):
            with h5py.File(path_to_hdf, 'r') as file:

                dataset_name = 'Timestamps_ms'
                if dataset_name in file:
                    time_stamps_ms = file[dataset_name]
                    time_stamps_seconds = []
                    for frame, time_stamp in enumerate(time_stamps_ms[:]):
                        time_stamps_seconds.append((frame, elapsed_time(reference_timestamp_ms=file[dataset_name][0], current_timestamp_ms=time_stamp)))

                    return time_stamps_seconds

        def trigger_frames(time_stamps_and_frames, triggers):
            def find_closest(time_stamps, trigger):
                """Find the closest time stamp to the trigger."""
                closest = None
                min_diff = float('inf')
                for ts in time_stamps:
                    diff = abs(ts[1] - trigger)
                    if diff < min_diff:
                        min_diff = diff
                        closest = ts[0]
                return closest

            prepared_time_stamps = []

            for ts in time_stamps_and_frames:
                try:
                    prepared_ts = (ts[0], float(ts[1].rstrip('.')))
                except ValueError:
                    prepared_ts = (ts[0], 0.0)
                prepared_time_stamps.append(prepared_ts)

            trigger_frames = [find_closest(prepared_time_stamps, triggers) for triggers in triggers]

            # new code #

            def find_closest_frame(time_stamps, target_frame):
                """
                Find the frame number closest to the given frame, considering the offset in milliseconds.
                """
                # Convert the offset to seconds
                offset_seconds_before = 50 / 1000
                offset_seconds_after = 100 / 1000

                # Get the time for the target frame
                target_time = next((time for frame, time in time_stamps if frame == target_frame), None)
                if target_time is None:
                    return None

                # Calculate the target times with offset
                target_time_before = target_time - offset_seconds_before
                target_time_after = target_time + offset_seconds_after

                # Find the closest frames to the target times
                closest_before = min(time_stamps, key=lambda x: abs(x[1] - target_time_before))[0]
                closest_after = min(time_stamps, key=lambda x: abs(x[1] - target_time_after))[0]

                return closest_before, closest_after

            def create_result_list(trigs, time_stamps):
                result = []
                for frame in trigs:
                    before, after = find_closest_frame(time_stamps, frame)
                    result.append((before, frame, after))
                return result

            trigger_pairs = create_result_list(trigs=trigger_frames, time_stamps=prepared_time_stamps)

            return trigger_pairs

        triggers = read_numbers_from_file(file_path=self.path_to_time_stamps_file)
        tuples_with_frames_and_time_stamps = read_hdf_file(path_to_hdf=self.path_to_hdf)
        trigger_frames_pairs = trigger_frames(time_stamps_and_frames=tuples_with_frames_and_time_stamps, triggers=triggers)
        return trigger_frames_pairs

    @staticmethod
    def faces(path_to_text_file):
        file = open(path_to_text_file, 'r')
        faces_ = []
        for line in file.readlines():
            fname = line.rstrip().split(',')
            faces_.append(fname)

        faces_ = [item for sublist in faces_ for item in sublist]

        def replace_value(value):
            if value == '2':
                return 'n'
            elif value == '1':
                return 'f'
            elif value == '0':
                return 'i'
            else:
                return value

        faces_ = [(replace_value(value), index+1) for index, value in enumerate(faces_)]

        return faces_

    @staticmethod
    def concatenate_videos(subject_name_block_name, path_to_subject_folder, output_folder):

        path = path_to_subject_folder

        video_paths = glob.glob(f'{path}./*.mp4')

        audio_triggers_files = [s for s in video_paths if "_aud_" in s]
        visual_triggers_files = [s for s in video_paths if "_vis_" in s]

        for trial, (visual_trigger, audio_trigger,) in enumerate(zip(visual_triggers_files, audio_triggers_files, )):
            audio_trigger_ = VideoFileClip(audio_trigger)
            audio_trigger_.set_fps(120)

            visual_trigger_ = VideoFileClip(visual_trigger)
            visual_trigger_.set_fps(120)

            combined = clips_array([[visual_trigger_], [audio_trigger_], ])
            combined.write_videofile(
                fr"{output_folder}\{subject_name_block_name}_{trial}.mp4")


