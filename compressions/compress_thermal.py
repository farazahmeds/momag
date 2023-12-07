import h5py
import time
from numba import jit
import glob
import os

def find_h5_files(folder_path):
    search_pattern = os.path.join(folder_path, '*.h5')

    h5_files = glob.glob(search_pattern)

    return h5_files

files = find_h5_files(folder_path=r'E:\subject_3\thermal')

for i, file in enumerate(files):
    compressed_file_path = rf'F:\dils_compressed\subject_3\thermal\subject_3_{i}.h5'

    start = time.perf_counter()

    with h5py.File(file, 'r') as original_file:
        with h5py.File(compressed_file_path, 'w') as compressed_file:
            @jit(nopython=False)
            def process_data():
                for dataset_name in original_file:
                    dataset = original_file[dataset_name]
                    total_frames = dataset.shape[0]
                    chunk_size = 100
                    for start_idx in range(0, total_frames, chunk_size):
                        end_idx = min(start_idx + chunk_size, total_frames)
                        data_chunk = dataset[start_idx:end_idx]
                        if dataset_name not in compressed_file:
                            maxshape = list(dataset.shape)
                            maxshape[0] = None
                            if dataset_name == 'Frames':
                                compressed_file.create_dataset(dataset_name, data=data_chunk, maxshape=maxshape, compression='gzip', compression_opts=6, chunks=(1,dataset.shape[1], dataset.shape[2]))
                            else:
                                compressed_file.create_dataset(dataset_name, data=data_chunk, maxshape=maxshape,
                                                               compression='gzip', compression_opts=6)

                        else:
                            compressed_file[dataset_name].resize((compressed_file[dataset_name].shape[0] + data_chunk.shape[0]), axis=0)
                            compressed_file[dataset_name][-data_chunk.shape[0]:] = data_chunk

            process_data()

    end = time.perf_counter()
    print(end - start)
    print("Compression completed.")
