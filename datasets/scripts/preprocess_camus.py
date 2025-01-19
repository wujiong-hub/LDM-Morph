import glob
import nibabel as nib
import SimpleITK as sitk
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
import os

def scale(imgpath, maskpath):
    org_img = nib.load(imgpath)
    gt_img = nib.load(maskpath)
    org_arr = np.transpose(org_img.get_fdata()[::-1])
    gt_arr = np.transpose(gt_img.get_fdata()[::-1])

    desired_size = (112,112)
    zoom_factors = [d/o for d, o in zip(desired_size, org_arr.shape)]
    resampled_org = scipy.ndimage.zoom(org_arr, zoom_factors, order=3)
    resampled_gt = scipy.ndimage.zoom(gt_arr, zoom_factors, order=0)

    img = np.zeros((128,128), np.float32)
    lab = np.zeros((128,128), np.float32)
    img[8:-8,8:-8] = resampled_org
    lab[8:-8,8:-8] = resampled_gt

    return img, lab


def imgshow(resampled_org, resampled_gt):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(resampled_org, cmap='gray', interpolation='none')
    axes[0].set_title('Original Image0')
    axes[0].axis('off')  # Hide the axis

    axes[1].imshow(resampled_org, cmap='gray', interpolation='none')
    axes[1].imshow(resampled_gt, cmap='Blues', alpha=0.2, interpolation='none')
    axes[1].set_title('Original Image with Mask Overlay')
    axes[1].axis('off')  # Hide the axis

    axes[2].imshow(resampled_gt, cmap='gray', interpolation='none')
    axes[2].set_title('Original Image1')
    axes[2].axis('off')  # Hide the axis

    plt.tight_layout()
    plt.show()


all2ched = glob.glob('../CAMUS/database_nifti/patient*/patient*_2CH_ED.nii.gz')
all2ches = glob.glob('../CAMUS/database_nifti/patient*/patient*_2CH_ES.nii.gz')
all4ched = glob.glob('../CAMUS/database_nifti/patient*/patient*_4CH_ED.nii.gz')
all4ches = glob.glob('../CAMUS/database_nifti/patient*/patient*_4CH_ES.nii.gz')

all2ched_gt = glob.glob('../CAMUS/database_nifti/patient*/patient*_2CH_ED_gt.nii.gz')
all2ches_gt = glob.glob('../CAMUS/database_nifti/patient*/patient*_2CH_ES_gt.nii.gz')
all4ched_gt = glob.glob('../CAMUS/database_nifti/patient*/patient*_4CH_ED_gt.nii.gz')
all4ches_gt = glob.glob('../CAMUS/database_nifti/patient*/patient*_4CH_ES_gt.nii.gz')

all2ched.sort()
all2ches.sort()
all4ched.sort()
all4ches.sort()

all2ched_gt.sort()
all2ches_gt.sort()
all4ched_gt.sort()
all4ches_gt.sort()

for i in range(len(all2ched)):
    savename2ch = os.path.basename(all2ches[i]).split('_E')[0]+'.npz'
    savename4ch = os.path.basename(all4ches[i]).split('_E')[0]+'.npz'
    print(savename2ch, savename4ch)
    
    ched2, ched2_gt = scale(all2ched[i], all2ched_gt[i])
    ches2, ches2_gt = scale(all2ches[i], all2ches_gt[i])
    ched4, ched4_gt = scale(all4ched[i], all4ched_gt[i])
    ches4, ches4_gt = scale(all4ches[i], all4ches_gt[i])

    if i==10:
        imgshow(ches2, ches2_gt)
        imgshow(ched2, ched2_gt)
        imgshow(ches4, ches4_gt)
        imgshow(ched4, ched4_gt)

    np.savez(os.path.join('../CAMUS/prep/',savename2ch), ES_img=ches2, ES_lab=ches2_gt, \
                                                    ED_img=ched2, ED_lab=ched2_gt)
    np.savez(os.path.join('../CAMUS/prep/',savename4ch), ES_img=ches4, ES_lab=ches4_gt, \
                                                    ED_img=ched4, ED_lab=ched4_gt)
