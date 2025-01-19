import SimpleITK as sitk
import nibabel as nib
import glob, os
import numpy as np
from scipy.ndimage import zoom

def resample_image_to_specific_spacing(image_path, new_spacing=[1.0, 1.0, 1.0], useNearest=False):
    # Load the image
    image = sitk.ReadImage(image_path)
    
    # Get the original size and spacing
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    
    # Calculate the new size, maintaining the same volume (approximately)
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]
    
    # Set up the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    if useNearest:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    
    # Perform the resampling
    resampled_image = resampler.Execute(image)
    
    return resampled_image


def resample(imgpath, savepath):
    allsubjects = glob.glob(imgpath)
    allsubjects.sort()
    for isub in allsubjects:
        allimgs = glob.glob(isub+'*_frame*.nii.gz')
        allimgs.sort()

        for i_img in allimgs:
            img = sitk.ReadImage(i_img)
            imgsize = img.GetSize()
            savename =  os.path.join(savepath,i_img.split('/')[-1])

            if imgsize[0]>imgsize[1] and ('patient141' not in i_img) and ('patient117' not in i_img) and ('patient126' not in i_img) and ('patient143' not in i_img):
                print(imgsize)
                print(i_img)
                nibimg = nib.load(i_img)
                nibarr = nibimg.get_fdata()
                nibarr_ch = np.transpose(nibarr, (1,0,2))
                nibarr_ch = nibarr_ch[:,::-1,:]
                nibarr_chimg = nib.Nifti1Image(nibarr_ch, nibimg.affine)

                nib.save(nibarr_chimg, savename)
                img_ch = sitk.ReadImage(savename)
                img_ch.SetDirection(img.GetDirection())
                img_ch.SetOrigin(img.GetOrigin())
                img_ch.SetSpacing(img.GetSpacing())
                sitk.WriteImage(img_ch, savename)

            elif 'patient124' in i_img:
                nibimg = nib.load(i_img)
                nibarr = nibimg.get_fdata()
                nibarr_ch = np.transpose(nibarr, (1,0,2))
                nibarr_ch = nibarr_ch[:,::-1,:]
                nibarr_chimg = nib.Nifti1Image(nibarr_ch, nibimg.affine)

                nib.save(nibarr_chimg, savename)
                img_ch = sitk.ReadImage(savename)
                img_ch.SetDirection(img.GetDirection())
                img_ch.SetOrigin(img.GetOrigin())
                img_ch.SetSpacing(img.GetSpacing())
                sitk.WriteImage(img_ch, savename)
            
            else:
                sitk.WriteImage(img, savename)

            
            # Example usage
            # Adjust 'new_spacing' to your desired values
            if '_gt' in savename:
                resampled_image = resample_image_to_specific_spacing(savename, new_spacing=[1.5, 1.5, 3.15], useNearest=True)
            else:
                resampled_image = resample_image_to_specific_spacing(savename, new_spacing=[1.5, 1.5, 3.15])

            # To save the resampled image
            sitk.WriteImage(resampled_image, savename)


def crop(imgA_path, imgB_path, labA_path, labB_path):
    
    dataA = nib.load(imgA_path).get_fdata()
    dataB = nib.load(imgB_path).get_fdata()
    label_dataA = nib.load(labA_path).get_fdata()
    label_dataB = nib.load(labB_path).get_fdata()
    fineSize = [128, 128, 32]

    dataA -= dataA.min()
    dataA /= dataA.std()
    dataA -= dataA.min()
    dataA /= dataA.max()

    dataB -= dataB.min()
    dataB /= dataB.std()
    dataB -= dataB.min()
    dataB /= dataB.max()

    nh, nw, nd = dataA.shape
    sh = int((nh - fineSize[0]) / 2)
    sw = int((nw - fineSize[1]) / 2)
    dataA = dataA[sh:sh + fineSize[0], sw:sw + fineSize[1]]
    dataB = dataB[sh:sh + fineSize[0], sw:sw + fineSize[1]]
    label_dataA = label_dataA[sh:sh + fineSize[0], sw:sw + fineSize[1]]
    label_dataB = label_dataB[sh:sh + fineSize[0], sw:sw + fineSize[1]]

    if nd >= 32:
        sd = int((nd - fineSize[2]) / 2)
        dataA = dataA[..., sd:sd + fineSize[2]]
        dataB = dataB[..., sd:sd + fineSize[2]]
        label_dataA = label_dataA[..., sd:sd + fineSize[2]]
        label_dataB = label_dataB[..., sd:sd + fineSize[2]]
    else:
        sd = int((fineSize[2] - nd) / 2)
        dataA_ = np.zeros(fineSize)
        dataB_ = np.zeros(fineSize)
        dataA_[:, :, sd:sd + nd] = dataA
        dataB_[:, :, sd:sd + nd] = dataB
        label_dataA_ = np.zeros(fineSize)
        label_dataB_ = np.zeros(fineSize)
        label_dataA_[:, :, sd:sd + nd] = label_dataA
        label_dataB_[:, :, sd:sd + nd] = label_dataB
        dataA, dataB = dataA_, dataB_
        label_dataA, label_dataB = label_dataA_, label_dataB_
    
    desired_shape = (112, 112, dataA.shape[2])
    zoom_factors = [desired_shape[i] / dataA.shape[i] for i in range(len(desired_shape))]

    dataA = zoom(dataA, zoom_factors, order=3)
    label_dataA = zoom(label_dataA, zoom_factors, order=0)
    dataB = zoom(dataB, zoom_factors, order=3)
    label_dataB = zoom(label_dataB, zoom_factors, order=0)

    mid_indx = int(dataA.shape[2]/2)
    aslice = dataA[:,:,mid_indx]; bslice = dataB[:,:,mid_indx]
    alslice = label_dataA[:,:,mid_indx]; blslice = label_dataB[:,:,mid_indx]

    arra = np.zeros((128, 128), float); arrb = np.zeros((128, 128), float)
    arral = np.zeros((128, 128), float); arrbl = np.zeros((128, 128), float)
    arra[8:-8,8:-8] = (aslice-aslice.min())/(aslice.max()-aslice.min())
    arrb[8:-8,8:-8] = (bslice-bslice.min())/(bslice.max()-bslice.min())
    arral[8:-8,8:-8] = alslice ; arrbl[8:-8,8:-8] = blslice 


    #imgA, imgB = nib.Nifti1Image(arra, nib.load(imgA_path).affine), nib.Nifti1Image(arrb, nib.load(imgB_path).affine)
    #labA, labB = nib.Nifti1Image(arral, nib.load(labA_path).affine), nib.Nifti1Image(arrbl, nib.load(labB_path).affine)

    #savename = os.path.join('../slices/testing/', os.path.basename(imgA_path).split('_')[0]+'_'+str(j).zfill(2)+'.npz')


    return arra, arrb, arral, arrbl


if __name__ == '__main__':
    
    dataset_path = '../ACDC/database/*/*/'
    savepath = '../ACDC/prep/'

    allnames = sorted(glob.glob(savepath+'*.nii.gz'))
    allimgs = [i for i in allnames if 'gt' not in i]
    alllabs = [i for i in allnames if 'gt' in i]
    for i in range(int(len(allimgs)/2)):
        print(allimgs[i*2], allimgs[i*2+1], alllabs[i*2], alllabs[i*2+1])
        imgA, imgB, labA, labB = crop(allimgs[i*2], allimgs[i*2+1], alllabs[i*2], alllabs[i*2+1])

        savename = os.path.join(savepath, os.path.basename(allimgs[i*2]).split('_')[0]+'.npz')
        np.savez(savename, ES_img=imgA, ES_lab=labA, \
                                                 ED_img=imgB, mask_small=labB)

    
