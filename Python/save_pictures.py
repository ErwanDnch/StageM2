import numpy as np
import os
import time
import datetime
import gc
import matplotlib
from PIL import Image, ImageOps

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_image(file, dir_out, u_max):
    array = np.load(file)

    out_file = ''.join(''.join(file.split('/')[-1]).split('.')[0:-1])+'.png'
    out_path = f"{dir_out}/{out_file}"

    fig, ax = plt.subplots(dpi=148)  # Set dpi to match pixel size

    ax.imshow(array[:,:,0], vmin = 0.0, vmax = u_max, cmap='viridis', interpolation='nearest')
    ax.set_axis_off()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=148)
    plt.close(fig)
    del array
    gc.collect()


def save_image_PIL(file, dir_out, u_max):
    L0 = 1.72
    offx = 0.5
    array = np.load(file, mmap_mode='r')
    normalized = (array[:, :, 0] / u_max * 255).clip(0, 255).astype(np.uint8)

    #lenx = len(array)
    #
    #lenx = len(array)
    #lenf = int(2 * lenx / L0 * offx)
    #
    ## Normalize for colormap (0..1 float)
    #norm_float = normalized / 255.0
    #
    ## Apply colormap
    #cmap = plt.get_cmap('viridis')
    #colored = cmap(norm_float)  # NxMx4 RGBA floats
    #
    ## Convert to 8-bit RGB
    #colored_rgb = (colored[int(lenx/2-lenf/2):int(lenx/2+lenf/2), :lenf, :3] * 255).astype(np.uint8)
    #img = Image.fromarray(colored_rgb)


    # Normalize for colormap (0..1 float)
    norm_float = normalized / 255.0

    # Apply colormap
    cmap = plt.get_cmap('viridis')
    colored = cmap(norm_float)  # NxMx4 RGBA floats

    # Convert to 8-bit RGB
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(colored_rgb)


    out_file = os.path.splitext(os.path.basename(file))[0] + '.png'
    img.save(os.path.join(dir_out, out_file))
    del img, array
    gc.collect()


def get_not_done_files(out_dir, allfiles):
    done = {os.path.splitext(f)[0] for f in os.listdir(out_dir) if f.endswith(".png")}
    not_done = [f for f in allfiles if os.path.splitext(os.path.basename(f))[0] not in done]
    print(f"{len(not_done)} files to convert")
    return not_done

def remove_last_edit(dir):
    highest_mod_time = 0
    if len(os.listdir(dir))>0:
        for file in os.listdir(dir):
            mod_time = os.path.getmtime(os.path.join(dir,file))
            if highest_mod_time == 0 or mod_time > highest_mod_time:
                highest_mod_time = mod_time
                file_to_delete = file
    else: return 0
    print(datetime.datetime.fromtimestamp(highest_mod_time))
    print(file_to_delete)
    spl_file = file_to_delete.split(".")
    """
    if spl_file[-1] == "png":
        other_file = ".".join([spl_file[0],"npy"])
        dir_alt = os.path.join(dir,"numpy")
    else:
        other_file = ".".join([spl_file[0],"png"])
        dir_alt = os.path.join(dir,"images")
    """
    #print(f"Also: {other_file} {datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(dir_alt,other_file)))}")
    os.remove(os.path.join(dir,file_to_delete))
    #os.remove(os.path.join(dir_alt,other_file))
    return 0


def get_max_U_and_magnitude(files):
    U_max = 0

    for i, f in enumerate(files):
        array = np.load(f, mmap_mode='r')

        if array.shape[2] == 4:

            temp_array = np.sqrt(array[:,:,0]**2 + array[:,:,1]**2)
            U_max = max(U_max, np.max(temp_array))

            if np.max(temp_array)>U_max:
                U_max = np.max(temp_array)
                print(f)
                print(U_max)
            del temp_array

        elif np.max(array[5])>U_max:
            U_max = np.max(array[5])
            print(f)
            print(U_max)

    return U_max


src_directory = "/home/erwan/Master/Stage/SlipFromSim/data/results_U0.24_D0.1_N1024_LEVEL9/numpy"
out_directory = "/home/erwan/Master/Stage/SlipFromSim/data/results_U0.24_D0.1_N1024_LEVEL9/images"

files = []

if not os.path.isdir(out_directory):

    os.mkdir(out_directory)
    files = [os.path.join(src_directory, f) for f in os.listdir(src_directory) if os.path.isfile(os.path.join(src_directory, f))]
else:
    remove_last_edit(out_directory)
    allfiles = [os.path.join(src_directory, f) for f in os.listdir(src_directory) if os.path.isfile(os.path.join(src_directory, f))]
    files = get_not_done_files(os.path.join(out_directory), allfiles)



if __name__ == "__main__":
    lenlist = len(files)
    u_max = get_max_U_and_magnitude(files)

    time_start = time.time()

    for i,f in enumerate(files):
        save_image_PIL(f, out_directory, 0.5)
        per = (i+1)/lenlist*100
        s_rem = (time.time()-time_start)/(per)*(100-per)
        print(f"{per:.2f}%, time remaining: {s_rem//60:.0f}min {s_rem%60:.2f}s")
        for obj in gc.garbage:
            print(repr(obj))
