import pathlib
import time
import glob
import os
time_stamp_files = glob.glob(r'C:\Users\faraz\PycharmProjects\momag_fork\momag\momag_gui\time_stamps\Data./*.txt')

for files in time_stamp_files:
    time_stamps = []

    with open(f'{files}') as file:
        lines = [line.rstrip() for line in file]
    time_stamps.append(os.path.basename(os.path.normpath(files)))

    for line in lines:
        milliseconds = int((float(line) % 1) * 1000)

        formatted_time = time.strftime("%M:%S", time.gmtime(float(line)))

        formatted_time_with_milliseconds = f"{formatted_time}.{milliseconds:03d}"

        time_stamps.append(formatted_time_with_milliseconds)



    print (time_stamps)

    # with open("converted_time_stamps.txt", "w") as output:
    #     for items in time_stamps:
    #         output.write(str(time_stamps))













