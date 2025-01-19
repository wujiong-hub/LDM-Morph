import os
import collections
import pandas
import echonet
import numpy as np
import skimage.draw
import nibabel as nib
import matplotlib.pyplot as plt

def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)




#change the directory the echo video detaset saved
## please put the Videos and FileList.csv inside this root folder 
root = '../ECHO/'

with open(os.path.join(root, "FileList.csv")) as f:
    data = pandas.read_csv(f)
    data["Split"].map(lambda x: x.upper())

#print(data)

header = data.columns.tolist()
fnames = data["FileName"].tolist()
fnames = [fn + ".avi" for fn in fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix
outcome = data.values.tolist()

#print(outcome)

# Check that files are present
missing = set(fnames) - set(os.listdir(os.path.join(root, "Videos")))
if len(missing) != 0:
    print("{} videos could not be found in {}:".format(len(missing), os.path.join(root, "Videos")))
    for f in sorted(missing):
        print("\t", f)
    raise FileNotFoundError(os.path.join(root, "Videos", sorted(missing)[0]))

# Load traces
frames = collections.defaultdict(list)
trace = collections.defaultdict(_defaultdict_of_lists)

with open(os.path.join(root, "VolumeTracings.csv")) as f:
    header = f.readline().strip().split(",")
    assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

    for line in f:
        filename, x1, y1, x2, y2, frame = line.strip().split(',')
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        frame = int(frame)
        if frame not in trace[filename]:
            frames[filename].append(frame)
        trace[filename][frame].append((x1, y1, x2, y2))
for filename in frames:
    for frame in frames[filename]:
        trace[filename][frame] = np.array(trace[filename][frame])

# A small number of videos are missing traces; remove these videos
keep = [len(frames[f]) >= 2 for f in fnames]
fnames = [f for (f, k) in zip(fnames, keep) if k]
outcome = [f for (f, k) in zip(outcome, keep) if k]

#print(trace)

for index in range(len(fnames)):
    video = os.path.join(root, "Videos", fnames[index])

    video = echonet.utils.loadvideo(video).astype(np.float32)
    
    key = fnames[index]

    target = []
    for t in [ "Filename", "LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]:
        #key = fnames[index]
        if t == "Filename":
            target.append(fnames[index])
        elif t == "LargeFrame":
            target.append(video[:, frames[key][-1], :, :])
        elif t == "SmallFrame":
            target.append(video[:, frames[key][0], :, :])
        elif t in ["LargeTrace", "SmallTrace"]:
            if t == "LargeTrace":
                t = trace[key][frames[key][-1]]
            else:
                t = trace[key][frames[key][0]]
            x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
            x = np.concatenate((x1[1:], np.flip(x2[1:])))
            y = np.concatenate((y1[1:], np.flip(y2[1:])))

            r, c = skimage.draw.polygon(np.rint(y).astype(np.int32), np.rint(x).astype(np.int32), (video.shape[2], video.shape[3]))
            mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
            mask[r, c] = 1
            target.append(mask)

    #print(key, len(target))
    print(key, video.shape, target[1].shape, target[2].shape, target[3].shape, target[4].shape)
    
    imglargepadd = np.zeros((128,128), np.float32)
    lablargepadd = np.zeros((128,128), np.float32)
    imgsmallpadd = np.zeros((128,128), np.float32)
    labsmallpadd = np.zeros((128,128), np.float32)

    imglargepadd[8:-8,8:-8] = target[1][1,:,:]
    lablargepadd[8:-8,8:-8] = target[3]
    imgsmallpadd[8:-8,8:-8] = target[2][1,:,:]
    labsmallpadd[8:-8,8:-8] = target[4]

    np.savez(os.path.join('../Echo/prep/',target[0].split('.')[0]+'.npz'), ES_img=imglargepadd, ES_lab=lablargepadd, \
                                                 ED_img=imgsmallpadd, ED_lab=labsmallpadd)
    

    ## you can use the following codes to check the generated images and corresponding masks
    '''
    if index == 20:
        
        fig, axes = plt.subplots(1, 4, figsize=(10, 5))

        # Display the original image in the first subplot
        axes[0].imshow(imglargepadd, cmap='gray', interpolation='none')
        axes[0].set_title('Original Image0')
        axes[0].axis('off')  # Hide the axis

        axes[1].imshow(imglargepadd, cmap='gray', interpolation='none')
        axes[1].imshow(lablargepadd, cmap='Blues', alpha=0.2, interpolation='none')
        axes[1].set_title('Original Image with Mask Overlay')
        axes[1].axis('off')  # Hide the axis

        # Display the mask image in the second subplot
        axes[2].imshow(imgsmallpadd, cmap='gray', interpolation='none')
        axes[2].set_title('Original Image1')
        axes[2].axis('off')  # Hide the axis

        axes[3].imshow(imgsmallpadd, cmap='gray', interpolation='none')
        axes[3].imshow(labsmallpadd, cmap='Blues', alpha=0.2, interpolation='none')
        axes[3].set_title('Original Image with Mask Overlay')
        axes[3].axis('off')  # Hide the axis

        # Adjust layout
        plt.tight_layout()
        #plt.show()
        plt.savefig(target[0].split('.')[0]+'.png') 
        
        #np.savez(target[0].split('.')[0]+'.npz', img_large=target[1][1,:,:], mask_large=target[3], \
        #                                         img_small=target[2][1,:,:], mask_small=target[4])
        # 

    '''
