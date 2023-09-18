# This code finds out the list of available images names for which we want to keep the training data (in taining file created by assign_official_classes.py)
# The data for which we don't have images is discarded

import glob
import pandas as pd

img_list = list()
df = pd.DataFrame()

# Extracting all the files recursively from the directory for user 1
files = glob.glob("NTCIR14_u1_images/u1/**/*.JPG", recursive=True)

# Extracting file names from image_path
for file in files:
   extract = file.split("\\")
   ext_index = len(extract)
   image = extract[ext_index-2] +"/" + extract[ext_index-1];
   img_list.append(image)

# Using list to create a dataframe
df.insert(0, 'image_path', img_list)

# Adding the file names in an excel sheet
df.to_excel("TempData/Image_Listing_U1.xlsx", sheet_name='Image_Listing', index=False)

# Matching the image listing file with the training data created from code assign_official_classes.py
f1 = pd.read_excel("TempData/training_data_with_classes.xlsx")
f2 = pd.read_excel("TempData/Image_Listing_U1.xlsx")

# Merging the files - Inner join on imagepath column
f3 = f1.merge(f2[["image_path"]], on = "image_path", how = "inner")

# training_data files contain the data for which images are available
f3.to_excel("Dataset/training_data.xlsx", index = False)


