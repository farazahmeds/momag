import h5py
from numba import jit
import time

compressed_file_path = 'Kompress_Chunked.h5'
uncompressed_file_path = 'chunked.h5'

start = time.perf_counter()

with h5py.File(compressed_file_path, 'r') as compressed_file:
    with h5py.File(uncompressed_file_path, 'w') as uncompressed_file:
        @jit(nopython=False)
        def process_data():
            for dataset_name in compressed_file:
                dataset = compressed_file[dataset_name]
                total_frames = dataset.shape[0]
                chunk_size = 100
                for start_idx in range(0, total_frames, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_frames)
                    data_chunk = dataset[start_idx:end_idx]
                    if dataset_name not in uncompressed_file:
                        uncompressed_file.create_dataset(dataset_name, data=data_chunk, maxshape=(None,) + dataset.shape[1:])
                    else:
                        uncompressed_file[dataset_name].resize((uncompressed_file[dataset_name].shape[0] + data_chunk.shape[0]), axis=0)
                        uncompressed_file[dataset_name][-data_chunk.shape[0]:] = data_chunk

        process_data()

end = time.perf_counter()
print(end - start)
print("Uncompression completed.")
