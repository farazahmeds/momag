import h5py

with h5py.File('Kompress_Chunked.h5', 'r') as file:

    # List all the groups and datasets in the file
    print("Groups:")
    for group in file.keys():
        print(group)

    print("\nDatasets:")
    for dataset in file:
        print(dataset)

    # Read a specific dataset
    # dataset_name = 'Frames'
    # if dataset_name in file:
    #     data = file[dataset_name] # Read the entire dataset into memory, data = file[dataset_name][10] read only 10th fr
