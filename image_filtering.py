#importing required libraries
import cv2
import numpy as np
import glob
import os
import pandas as pd


def blurriness_homogeneity_check(img):
    # covert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # use of 3X3 laplacian filter on the gray image to calculate the variance
    blurriness = cv2.Laplacian(gray, cv2.CV_64F).var()
    # creation of quantized histogram using hue with 8 bins
    hist = cv2.calcHist([gray], [0], None, [8], [0, 180])
    quantized_hist = np.concatenate((hist), axis=0)
    # calcuating the percentage of homogeneity
    homogeneity = (max(quantized_hist) * 100) / sum(quantized_hist)
    # Thresholds decided based on trial and error method for NTCIR-14 dataset
    if (blurriness < 10 or homogeneity > 90):
        return 'Blurry/Homogeneous'
    else:
        return 'Not blurry/homogeneous'



#root folder to store image paths that can iterate through directories and sub-directories
root_path="/Users/tysonpais/Documents/Uni Koblenz/ML Research Lab/Implementation/u1"
root_folder = glob.glob(root_path+"/**/*.jpg", recursive=True)+glob.glob(root_path+"/**/*.JPG", recursive=True)
# initiating a dictionary to save the results
#all_files = {}
accepted_images_folderpath=root_path+'/'+"NTCIR14_Images"
rejected_images_folderpath=root_path+'/'+"NTCIR14_Images_Rejected"

#creation of directories
if (not os.path.exists(accepted_images_folderpath) and not os.path.exists(rejected_images_folderpath)):
    os.mkdir(accepted_images_folderpath)
    os.mkdir(rejected_images_folderpath)
elif (os.path.exists(accepted_images_folderpath) and not os.path.exists(rejected_images_folderpath)):
    os.mkdir(rejected_images_folderpath)
elif (not os.path.exists(accepted_images_folderpath) and os.path.exists(rejected_images_folderpath)):
    os.mkdir(accepted_images_folderpath)

# looping through all the image paths
for file in root_folder:
    # read the image using cv2
    img = cv2.imread(file)
    # extract the file name
    path, filename = os.path.split(file)
    # call the function with read image as parameter and store the value in the dictionary
    value = blurriness_homogeneity_check(img)
    #all_files[filename] = value
    if value == 'Not blurry/homogeneous':
        # Store the valid images in the folder called NTCIR14_Images
        cv2.imwrite(accepted_images_folderpath + '/' + filename, img)
    else:
        cv2.imwrite(rejected_images_folderpath + '/' + filename, img)

# saving the results in the data frame so it can be merged with the other dataframes on ImageName
#df = pd.DataFrame(all_files.items(), columns=['ImageName', 'ImageFilter'])
#print(df)