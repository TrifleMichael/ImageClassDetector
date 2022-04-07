import os
import shutil
from PIL import Image

file=open("flickr_logos_27_dataset_query_set_annotation.txt")

for line in file:
    line_arr=line.split()
    if not os.path.exists("images/"+line_arr[1]):
        os.mkdir("images/"+line_arr[1])
    image = Image.open("flickr_logos_27_dataset_images/"+line_arr[0])
    new_image = image.resize((256, 256))
    new_image.save("images/"+line_arr[1]+"/"+line_arr[0])
    #shutil.copy("flickr_logos_27_dataset_images/"+line_arr[0],"images/"+line_arr[1]+"/"+line_arr[0])