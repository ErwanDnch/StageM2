import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime

def read_and_save_file(file, dir_out, shape=(148,148,5)):
    array = np.zeros(shape)
    df = pd.read_csv(file)

    df.drop(columns=["Points:2"])

    List_y = list(set(df["Points:0"]))
    List_yi = [i for i in range(len(List_y))]
    List_y.sort()

    List_x = list(set(df["Points:1"]))
    List_xi = [i for i in range(len(List_x))]
    List_x.sort()

    for i in df.index:
        xi = List_x.index(df.iloc[i]["Points:1"])
        yi = List_y.index(df.iloc[i]["Points:0"])
        array[xi,yi,0] = df.iloc[i]["u.x"]
        array[xi,yi,1] = df.iloc[i]["u.y"]
        array[xi,yi,2] = df.iloc[i]["p"]
        array[xi,yi,3] = df.iloc[i]["omega"]
        array[xi,yi,4] = np.sqrt(array[xi,yi,0]**2+array[xi,yi,1]**2)

    out_file = ''.join(''.join(file.split('/')[-1]).split('.')[0:-1])+'.npy'
    out_path = f"{dir_out}/numpy/{out_file}"
    #print(out_path)
    np.save(out_path, array)


    out_file = ''.join(''.join(file.split('/')[-1]).split('.')[0:-1])+'.png'
    out_path = f"{dir_out}/images/{out_file}"

    fig, ax = plt.subplots(dpi=148)  # Set dpi to match pixel size

    ax.imshow(array[:,:,4], cmap='viridis', interpolation='nearest')
    ax.set_axis_off()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=148)
    plt.close()


def remove_last_edit(dir, folder):
    fulldir = os.path.join(dir, folder)
    highest_mod_time = 0
    for file in os.listdir(fulldir):
        mod_time = os.path.getmtime(os.path.join(fulldir,file))
        if highest_mod_time == 0 or mod_time > highest_mod_time:
            highest_mod_time = mod_time
            file_to_delete = file
    print(datetime.datetime.fromtimestamp(highest_mod_time))
    print(file_to_delete)
    spl_file = file_to_delete.split(".")
    if spl_file[-1] == "png":
        other_file = ".".join([spl_file[0],"npy"])
        dir_alt = os.path.join(dir,"numpy")
    else:
        other_file = ".".join([spl_file[0],"png"])
        dir_alt = os.path.join(dir,"images")
    print(f"Also: {other_file} {datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(dir_alt,other_file)))}")
    os.remove(os.path.join(fulldir,file_to_delete))
    os.remove(os.path.join(dir_alt,other_file))
    return 0


def get_not_done_files(dir, allfiles, dir_data):
    files = allfiles.copy()
    for f in os.listdir(dir):
        if os.path.join(dir_data,".".join([f.split(".")[0],"csv"])) in files:
            files.pop(files.index(os.path.join(dir_data,".".join([f.split(".")[0],"csv"]))))

    print(f"{len(files)} files to convert")
    return files


src_directory = "/home/erwan/Master/Stage/SlipFromSim/data/test"
out_directory = "/home/erwan/Master/Stage/SlipFromSim/data/NSFlume90s"

files = []

if not os.path.isdir(out_directory):
    os.mkdir(out_directory)
    os.mkdir(out_directory+"/numpy")
    os.mkdir(out_directory+"/images")
    files = [os.path.join(src_directory, f) for f in os.listdir(src_directory) if os.path.isfile(os.path.join(src_directory, f))]
else:
    remove_last_edit(out_directory,"numpy")
    remove_last_edit(out_directory, "images")
    allfiles = [os.path.join(src_directory, f) for f in os.listdir(src_directory) if os.path.isfile(os.path.join(src_directory, f))]
    files = get_not_done_files(os.path.join(out_directory, "numpy"), allfiles, src_directory)



lenlist = len(files)

time_start = time.time()
for i,f in enumerate(files):
    read_and_save_file(f, out_directory, shape=(256,256,5))
    per = (i+1)/lenlist*100
    s_rem = (time.time()-time_start)/(per)*(100-per)
    print(f"{per}%, time remaining: {s_rem//60}min {s_rem%60}s")
