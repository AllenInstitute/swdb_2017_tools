import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt

def download_structure_slices(template, mask, filename='my_slices.zip', half_sagittal = True):
    """ 
    After having imported the MouseConnectivityCache, get the template volume under the variable template. 
    Then create a mask for your region of choice under the variable mask (must be a region that is on both
    hemispheres). Input these arguments to receive a zipfile containing slice images of your region of interest 
    in the rostral-caudal, medial-lateral, dorsal-ventral orientations. If your region is present in both 
    hemispheres, assign half_sagittal to be True. 
     
    """

    slices_zip = zipfile.ZipFile(filename, 'w')
        
    my_template = template.copy()
    my_template[mask == 0] = 0


    mask_indices = np.nonzero(mask)

    for i in range(mask_indices[0].min(),mask_indices[0].max()):
        plt.clf()
        plt.close()
        f, ax = plt.subplots(figsize=(18,24))
        ax.imshow(my_template[i, :, :], interpolation='none', cmap=plt.cm.gray)
        plt.savefig('axis0_%d.png' % i)
        slices_zip.write('axis0_%d.png' % i, compress_type = zipfile.ZIP_DEFLATED)
        os.remove('axis0_%d.png' % i)
        
    for i in range(mask_indices[1].min(),mask_indices[1].max()):
        plt.clf()
        plt.close()
        f, ax = plt.subplots(figsize=(18,24))
        ax.imshow(my_template[:, i, :], interpolation='none', cmap=plt.cm.gray)
        plt.savefig('axis1_%d.png' % i)
        slices_zip.write('axis1_%d.png' % i, compress_type = zipfile.ZIP_DEFLATED)
        os.remove('axis1_%d.png' % i)
    

    if half_sagittal:
        half_mask = mask[:,:,0:mask.shape[2]/2]
        mask_indices = np.nonzero(half_mask)
    
        
    for i in range(mask_indices[2].min(),mask_indices[2].max()):
        plt.clf()
        plt.close()
        f, ax = plt.subplots(figsize=(18,24))
        ax.imshow(my_template[:, :, i], interpolation='none', cmap=plt.cm.gray)
        plt.savefig('axis2_%d.png' % i)
        slices_zip.write('axis2_%d.png' % i, compress_type = zipfile.ZIP_DEFLATED)
        os.remove('axis2_%d.png' % i)
    
    slices_zip.close()import os
