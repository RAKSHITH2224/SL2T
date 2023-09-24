# import numpy as np
# import cv2
# import os
# import csv
# from image_processing import func
# if not os.path.exists("data2"):
#     os.makedirs("data2")
# if not os.path.exists("data2/train"):
#     os.makedirs("data2/train")
# if not os.path.exists("data2/test"):
#     os.makedirs("data2/test")
# path="data"
# path1 = "data2"
# a=['label']
#
# for i in range(64*64):
#     a.append("pixel"+str(i))
#
#
# #outputLine = a.tolist()
#
#
# label=0
# var = 0
# c1 = 0
# c2 = 0
#
# for (dirpath,dirnames,filenames) in os.walk(path):
#     for dirname in dirnames:
#         print(dirname)
#         for(direcpath,direcnames,files) in os.walk(path+"/"+dirname):
#        	    if not os.path.exists(path1+"/train/"+dirname):
#                 os.makedirs(path1+"/train/"+dirname)
#             if not os.path.exists(path1+"/test/"+dirname):
#                 os.makedirs(path1+"/test/"+dirname)
#             # num=0.75*len(files)
#             num = 100000000000000000
#             i=0
#             for file in files:
#                 var+=1
#                 actual_path=path+"/"+dirname+"/"+file
#                 actual_path1=path1+"/"+"train/"+dirname+"/"+file
#                 actual_path2=path1+"/"+"test/"+dirname+"/"+file
#                 img = cv2.imread(actual_path)
#                 bw_image = func(actual_path)
#                 if i<num:
#                     c1 += 1
#                     cv2.imwrite(actual_path1 , bw_image)
#                 else:
#                     c2 += 1
#                     cv2.imwrite(actual_path2 , bw_image)
#
#                 i=i+1
#
#         label=label+1
# print(var)
# print(c1)
# print(c2)
#
#
#





# import os
# import cv2
#
# # Set the path to your 'data' directory containing subdirectories
# data_directory = "data"
#
# # Create a directory to store the resulting grayscale images
# output_directory = "output_gray_images"
# os.makedirs(output_directory, exist_ok=True)
#
# # Function to process a single image
# def process_image(image_path, output_dir):
#     # Read the colored image
#     colored_image = cv2.imread(image_path)
#
#     # Convert the colored image to grayscale
#     gray_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
#
#     # Apply Gaussian blur to the grayscale image
#     blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 2)
#
#     # Get the relative path from the 'data' directory to the image
#     relative_path = os.path.relpath(image_path, data_directory)
#
#     # Construct the output path for the resulting image
#     output_path = os.path.join(output_dir, relative_path)
#
#     # Create the output directory if it doesn't exist
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     # Save the resulting grayscale image
#     cv2.imwrite(output_path, blurred_image)
#
# # Recursively process images in subdirectories
# for root, dirs, files in os.walk(data_directory):
#     for file in files:
#         if file.lower().endswith((".png", ".jpg", ".jpeg")):
#             image_path = os.path.join(root, file)
#             process_image(image_path, output_directory)
#             print(f"Processed: {image_path}")
#
# print("All images processed and saved.")













import os
import cv2

# Set the path to your 'data' directory containing subdirectories
data_directory = "data"

# Create a directory to store the resulting images
output_directory = "output_processed_images"
os.makedirs(output_directory, exist_ok=True)

# Function to process a single image
def process_image(image_path, output_dir):
    # Read the colored image
    colored_image = cv2.imread(image_path)

    # Convert the colored image to grayscale
    gray_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply adaptive thresholding to the grayscale image
    thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Get the relative path from the 'data' directory to the image
    relative_path = os.path.relpath(image_path, data_directory)

    # Construct the output path for the resulting image
    output_path = os.path.join(output_dir, relative_path)

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the resulting thresholded image
    cv2.imwrite(output_path, thresholded_image)

# Recursively process images in subdirectories
for root, dirs, files in os.walk(data_directory):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(root, file)
            process_image(image_path, output_directory)
            print(f"Processed: {image_path}")

print("All images processed and saved.")





