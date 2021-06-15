# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 00:08:53 2021

@author: maxence and aman
"""
import nibabel as nib
import numpy as np
import numpy.ma as ma
import os
from scipy.ndimage.measurements import label, center_of_mass
from scipy import ndimage
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu
from deepbrain import Extractor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import SimpleITK as sitk
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from scipy.ndimage import convolve, gaussian_filter
from scipy.stats import norm as normal
from scipy.ndimage.morphology import binary_fill_holes
from matplotlib import pyplot as plt

INTENSITY  = 1700#1820#
RANGE_DOWN = 200
RANGE_UP   = 500
MINIMUM_LESION_SIZE = 25


##############################################
# FUNCTIONS USED FOR THE CUSTOM SEGMENTATION #
##############################################


def make_brainmask(bids_dir, subject, save_mask=True, sequence='MPRAGE'):

    # Load a nifti as 3d numpy image [H, W, D]
    img = nib.load(os.path.join(bids_dir, 'derivatives', 'transformations',
                                f'sub-{subject}', 'ses-01',
                                f'sub-{subject}_ses-01_MPRAGE_reg-FLAIR.nii.gz'))
    mx = img.get_fdata()
    ext = Extractor()

    # `prob` will be a 3d numpy image containing probability
    # of being brain tissue for each of the voxels in `img`
    prob = ext.run(mx)

    # mask can be obtained as:
    mask = prob > 0.5

    mx[mask == False] = 0

    nifti_out = nib.Nifti1Image(mx, affine=img.affine)
    nib.save(nifti_out, os.path.join(bids_dir, "derivatives", "transformations",
                                     f"sub-{subject}", "ses-01",
                                      f"sub-{subject}_ses-01_{sequence}_brain.nii.gz"))

    if save_mask:
        mask = mask.astype(int)
        nifti_out = nib.Nifti1Image(mask, affine=img.affine)
        nib.save(nifti_out, os.path.join(bids_dir, "derivatives", "transformations",
                                         f"sub-{subject}", "ses-01",
                                          f"sub-{subject}_ses-01_{sequence}_brainmask.nii.gz"))



def apply_brainmask(bids_dir, subject, sequence='FLAIR'):
    img = nib.load(os.path.join(bids_dir, f"sub-{subject}", "ses-01", "anat",
                                      f"sub-{subject}_ses-01_{sequence}.nii.gz"))
    brain = img.get_fdata()

    mask = nib.load(os.path.join(bids_dir, "derivatives", "transformations",
                                 f"sub-{subject}", "ses-01",
                                 f"sub-{subject}_ses-01_MPRAGE_brainmask.nii.gz"))
    mask = mask.get_fdata()

    brain[mask == 0] = 0

    nifti_out = nib.Nifti1Image(brain, affine=img.affine)
    nib.save(nifti_out, os.path.join(bids_dir, "derivatives", "transformations",
                                     f"sub-{subject}", "ses-01",
                                     f"sub-{subject}_ses-01_{sequence}_brain.nii.gz"))


# register single patient
def mprage2flair_registration(bids_dir, subject, save_tfm = False):


    pathToFixed = os.path.join(bids_dir, f"sub-{subject}", "ses-01", "anat",
                                      f"sub-{subject}_ses-01_FLAIR.nii.gz")
    pathToMoving = os.path.join(bids_dir, f"sub-{subject}", "ses-01", "anat",
                                      f"sub-{subject}_ses-01_MPRAGE.nii.gz")
    registration_method = sitk.ImageRegistrationMethod()
    # FIRST STEP : nii to mha file (simpleITK u)
    sitk.WriteImage(sitk.ReadImage(pathToFixed), pathToFixed.replace(".nii.gz", "")+'.mha')   #read .nii.gz to write .mha
    sitk.WriteImage(sitk.ReadImage(pathToMoving), pathToMoving.replace(".nii.gz", "")+'.mha')
    fixed_image = sitk.ReadImage(pathToFixed.replace(".nii.gz", "")+'.mha', sitk.sitkFloat32) # read the new .mha file
    moving_image = sitk.ReadImage(pathToMoving.replace(".nii.gz", "")+'.mha', sitk.sitkFloat32)

    # SECOND STEP : map center of mass moving and fixed
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                      moving_image,
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # THIRD STEP : registrate

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,
                                                      numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6,
                                                      convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Always check the reason optimization terminated.
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    # FORTH STEP : export result
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform,
                                     sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    OUTPUT_DIR = os.path.join(bids_dir,'derivatives','transformations',
                              f'sub-{subject}','ses-01')
    sitk.WriteImage(moving_resampled,
                    os.path.join(OUTPUT_DIR, f'sub-{subject}_ses-01_MPRAGE_reg-FLAIR.nii.gz'))

    if save_tfm:
        sitk.WriteTransform(final_transform,
                            os.path.join(OUTPUT_DIR, f'sub-{subject}_ses-01_MPRAGE_reg-FLAIR.tfm'))



def make_gm_wm_mask(bids_dir, subject):

    sequence = 'MPRAGE'
    # Load images
    img   = nib.load(os.path.join(bids_dir, 'derivatives', 'transformations',
                                  f"sub-{subject}", "ses-01",
                                  f"sub-{subject}_ses-01_{sequence}_reg-FLAIR.nii.gz"))
    im = img.get_fdata()
    mask = nib.load(os.path.join(bids_dir, "derivatives", "transformations",
                                 f"sub-{subject}", "ses-01",
                                 f"sub-{subject}_ses-01_{sequence}_brainmask.nii.gz"))
    mask = mask.get_fdata()

    # Calculate Otsu threshold on entire image
    th = threshold_otsu(im)

    # Now do same for masked image
    masked = ma.masked_array(im,mask==0, fill_value=0)
    val = threshold_otsu(masked.compressed())


    mx = np.where((masked<val) & (mask!=0), 500, masked)
    mx = np.where((masked<th) & (mask!=0),  0,   mx)
    mx = np.where((masked>=val) & (mask!=0), 1000, mx)
    mx = np.where(mask == 0, 0, mx)

    mx = ndimage.median_filter(mx, footprint=np.ones((3,3,3)))

    nifti_out = nib.Nifti1Image(mx, affine=img.affine)
    nib.save(nifti_out, os.path.join(bids_dir, 'derivatives', 'segmentations',
                                     f"sub-{subject}", "ses-01",
                                     f"sub-{subject}_ses-01_gm_wm.nii.gz"))


def histogram_matching(bids_dir, subject):
    REFERENCE_FLAIR = os.path.join(bids_dir, "sub-002","ses-01","anat",
                                   "sub-002_ses-01_FLAIR_brain.nii.gz")


    flair_img = nib.load(os.path.join(bids_dir, "derivatives", "transformations",
                                      f"sub-{subject}", "ses-01",
                                      f"sub-{subject}_ses-01_FLAIR_brain.nii.gz"))
    flair = flair_img.get_fdata()

    ref_img = nib.load(REFERENCE_FLAIR)
    ref = ref_img.get_fdata()

    matched=np.zeros_like(flair)
    matched[flair>0] = match_histograms(flair[flair>0], ref[ref>0], multichannel=False)

    nifti_out = nib.Nifti1Image(matched, affine=flair_img.affine)
    nib.save(nifti_out, os.path.join(bids_dir, "derivatives", "transformations",
                                     f"sub-{subject}", "ses-01",
                                     f"sub-{subject}_ses-01_FLAIR_brain-matched.nii.gz"))


def compute_lesion_mask(bids_dir, subject, intensity=INTENSITY,
                        range_down=RANGE_DOWN, range_up=RANGE_UP, matched=False,
                        use_wm_gm=True):


    if matched:
        flair_img = nib.load(os.path.join(bids_dir, "derivatives", "transformations",
                                          f"sub-{subject}", "ses-01",
                                          f"sub-{subject}_ses-01_FLAIR_brain-matched.nii.gz"))
    else:
        flair_img = nib.load(os.path.join(bids_dir, "derivatives", "transformations",
                                          f"sub-{subject}", "ses-01",
                                          f"sub-{subject}_ses-01_FLAIR_brain.nii.gz"))

    flair = flair_img.get_fdata()

    mask = np.logical_and(flair > intensity-range_down, flair < intensity+range_up)
    lesion_mask = mask.astype(int)


    lesion_mask = ndimage.median_filter(lesion_mask, footprint=np.ones((3,3,3)))


    wm_gm_path = os.path.join(bids_dir, 'derivatives', 'segmentations',
                            f"sub-{subject}", "ses-01",
                            f"sub-{subject}_ses-01_gm_wm.nii.gz")

    if use_wm_gm:
        wm_gm_img = nib.load(wm_gm_path)
        wm_gm = wm_gm_img.get_fdata()

        labeled_lesions = lesion_mask
        structure = np.ones((3, 3, 3))
        labeled_lesions, nlesions = label(lesion_mask,structure)

        discarded = 0

        for lesion_id in range(1,nlesions+1):
            com = center_of_mass(lesion_mask, labels=labeled_lesions, index=lesion_id)
            com = (int(com[0]),int(com[1]),int(com[2]))
            print(f"Lesion {lesion_id}/{nlesions}: {com}", end = "> ")

            val = wm_gm[com]
            if val != 1000:
                print(f"discarded (val = {val})")
                lesion_mask[labeled_lesions == lesion_id] = 0
                labeled_lesions[labeled_lesions == lesion_id] = 0
                discarded += 1
            else:
                print("retained")

        print(f"Discarded lesions: {discarded}")



    nifti_out = nib.Nifti1Image(lesion_mask, affine=flair_img.affine)
    nib.save(nifti_out, os.path.join(bids_dir, "derivatives", "segmentations",
                                     f"sub-{subject}", "ses-01",
                                     f"sub-{subject}_ses-01_lesion-mask.nii.gz"))

    
####################################
# CLASSICAL METHODS IMPLEMENTATION #
####################################

# OTSU

def otsu(I, PMF):
    sigma = 0             # inter-class variance
    thresh = -1           # threshold found by algorithm
    sigma_prev = -1       # value of previous inter-class variance
    
    for t in range(1,256):
        w0 = sum(PMF[0:t])
        w1 = 1-w0
        u0 = sum([i*PMF[i] for i in range(0,t)])/w0
        u1 = sum([i*PMF[i] for i in range(t,256)])/(1-w0)
        sigma = w0*w1*(u0-u1)*(u0-u1)
        
        if(sigma > sigma_prev):
            thresh = t
            sigma_prev = sigma
    
    OutputIm = np.zeros((I.shape[0],I.shape[1])) # Output image after otsu algorithm
    
    for i in range(0,I.shape[0]):
        for j in range(0,I.shape[1]):
            if I[i,j] <= thresh:
                OutputIm[i,j] = 0
            else:
                OutputIm[i,j] = 255
            
    return OutputIm

# Expectation Maximization classification 

# Useful function for EM

def getForegroundMask(originalBrainImg, backGroundValue=0):
    # Get a binary mask from a 2D array, where the pixels > 0 are set to 1
    foregroundMask = originalBrainImg > backGroundValue
    return binary_fill_holes(foregroundMask)

def getForegroundArray(originalBrainImg, foregroundMask):
    # Return a 1D array with the pixels from originalBrainImg which are under the foregroundmask
    return originalBrainImg[foregroundMask==1]
 
def plotHistogramWithDistribs(array, meansArray, stdsArray):
    # Plot the histogram of an array and gaussian distributions corresponding to meansArray / stdsArray
    plt.figure(figsize=(10, 10))
    hist, bin_edges = np.histogram(array, bins=255, normed=True)
    plt.bar(bin_edges[:-1], hist, align='center', width=0.005)
    plt.ylabel('Number of Pixels')
    plt.xlabel('Intensity')
    x = np.linspace(0, 1, 255)
    for (mean, std) in zip(meansArray, stdsArray):
        plt.plot(x, normal.pdf(x, mean, std), linewidth=1)
        
# EM algorithm

def em(pixelArray, meansArray, stdsArray, priorProbs, tol=1e-4, max_iter=1000):
    
    # In order to check the stopping condition
    mean_diff = np.ones(len(priorProbs)) 
    std_diff = np.ones(len(priorProbs))
    
    # Cpy of the initial means and stds values for clean reuse in loop 
    MeansArray = meansArray
    StdsArray = stdsArray
    
    # updated values of arrays after iteration i
    newMeansArray = np.ones(len(priorProbs))
    newStdsArray = np.ones(len(priorProbs))
    
    # denominator of P(z_i = c_k | y_i, mu_k, sigma_k)
    denominator = np.zeros((1, len(pixelArray)))

    pixels_class_probs = np.zeros((len(priorProbs), len(pixelArray))) # P(z_i=c_k | y_i, mu_k, sigma_k), each row is a class
                                                        # and each col is a pixel
    P_yi = np.zeros((len(priorProbs), len(pixelArray))) # P(y_i|z_i = c_k, mu_k, sigma_k), each row is a class
                                                        # and each col is a pixel
    
    for iterNumber in range(max_iter):
        if (np.mean(mean_diff) < tol and np.mean(std_diff) < tol):
            break
        
        # Step E(xpectaction)
        
        # First compute P(y_i|z_i = c_k, mu_k, sigma_k) (equation 1)
        for class_index in range(0, len(priorProbs)):
            for pixel_index in range(0, len(pixelArray)):
                P_yi[class_index, pixel_index] =(1/(np.sqrt(2*np.pi)*StdsArray[class_index]))*np.exp(-0.5*np.square((pixelArray[pixel_index]-MeansArray[class_index])/StdsArray[class_index]))


        # Compute denominator of P(z_i = c_k | y_i, mu_k, sigma_k) to make it easier afterward
        for class_index in range(0, len(priorProbs)):
            P_yi[class_index,:] = P_yi[class_index,:]*priorProbs[class_index]
        for pixel_index in range(0, len(pixelArray)):
            denominator[0,pixel_index]= np.sum(P_yi[:,pixel_index]) # 1x#pixels

        # Compute class probabilites P(z_i = c_k | y_i, mu_k, sigma_k) (equation 2)
        for class_index in range(0, len(priorProbs)):
            for pixel_index in range(0, len(pixelArray)):
                pixels_class_probs[class_index, pixel_index] = (P_yi[class_index, pixel_index]*priorProbs[class_index])/denominator[0, pixel_index]


        # Step M(aximization)
        
        # update theta by maximizing likelihood (computed manually) therefore only update by formula given in answer above

        for class_index in range(0, len(priorProbs)):
            newMeansArray[class_index] = np.dot(pixels_class_probs[class_index,:],pixelArray)/np.sum(pixels_class_probs[class_index,:])
            num = 0 # numerator of std array
            for pixel_index in range(0, len(pixelArray)):
                num = num + pixels_class_probs[class_index,pixel_index] * np.square(pixelArray[pixel_index]-newMeansArray[class_index])
            newStdsArray[class_index] = np.sqrt(num/np.sum(pixels_class_probs[class_index,:]))
        
        # Update prior probabilities
        
        # Compute convergence conditions to check in next iteration
        mean_diff = np.abs(MeansArray - newMeansArray)
        std_diff = np.abs(StdsArray - newStdsArray)
        
        # Update the arrays 
        MeansArray = newMeansArray
        StdsArray = newStdsArray
        
    return pixels_class_probs, newMeansArray, newStdsArray

# REGION GROWING algorithm

# Mostly the same as pseudo code : 

# sel = 1 -> PD sel = 2 -> T1
def criterion(pixel, image, sel):
    
    if(sel == 0): #PD criterion
        if(image[pixel[0], pixel[1]] > 180):
            return True
        else:
            return False
        
    else: #T1 criterion
        if(image[pixel[0], pixel[1]] < 25):
            return True
        else:
            return False

def neighbours_not_checked(pixel, vector, checkedImage):
    
    
    # top pixel
    if(pixel[1]+1 > 255):
        print('outLimit') #do not check out of limit
    else:
        if(checkedImage[pixel[0], pixel[1]+1] == 0): # if not checked
            vector.append([pixel[0], pixel[1]+1]) # add to unchecked pixels
        
    # bottom pixel    
    if(pixel[1]-1 < 0):
        print('outLimit') #do not check out of limit
    else:
        if(checkedImage[pixel[0], pixel[1]-1] == 0):
            vector.append([pixel[0], pixel[1]-1])
        
    # right pixel
    if(pixel[0]+1 > 255):
        print('outLimit') #do not check out of limit
    else:
        if(checkedImage[pixel[0]+1, pixel[1]] == 0):
            vector.append([pixel[0]+1, pixel[1]])
        
    # left pixel
    if(pixel[0]-1 < 0):
        print('outLimit') #do not check out of limit
    else:
        if(checkedImage[pixel[0]-1, pixel[1]] == 0):
            vector.append([pixel[0]-1, pixel[1]])
        

def regionGrowing(imageList, seedPosList) :
    
    plt.figure()
    N = len(imageList)
    for i in range(0, N): # for each image and seed
        image = imageList[i]
        plt.subplot(N, N, i+1)
        plt.imshow(image, cmap='gray')

        seed_pixel = seedPosList[i] #(x,y) of image i
        regionImage = image.copy()
        checkedImage = np.zeros((256,256))
        regionImage[seed_pixel[0], seed_pixel[1]] = 1 # add to region 
        checkedImage[seed_pixel[0], seed_pixel[1]] = 1 # mark seed_pixel as visited
        vector = [] 
        neighbours_not_checked(seed_pixel, vector, checkedImage) # add uncheck pixel in the nei of seed_pixel in vector
        
        while len(vector) > 0 :
            current_pixel = vector[0]
            if(criterion(current_pixel, image, i) and checkedImage[current_pixel[0],current_pixel[1]] == 0):
                regionImage[current_pixel[0], current_pixel[1]] = 255
                neighbours_not_checked(current_pixel, vector, checkedImage)
                checkedImage[current_pixel[0], current_pixel[1]] = 1
                
            vector = vector[1:len(vector)]

        plt.subplot(N, N, (i+1)+N)
        plt.imshow(regionImage, cmap='gray')
        print(i)




