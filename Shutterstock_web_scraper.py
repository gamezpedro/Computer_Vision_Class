'''
Use it from terminal:

python3 Shutterstock_web_scraper.py <search_term> <number_of_images>

'''

import cv2
import os
import requests
import sys
import numpy as np
from PIL import Image
from io import BytesIO
from selenium import webdriver

def downloadImage(url, file_name):
    response = requests.get(url, stream=True)
    img = Image.open(BytesIO(response.content))
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, img)

driver = webdriver.Firefox()

# Create directories
to_search = sys.argv[1]
cwd = os.getcwd()
absolute_train_dir = os.path.join(cwd, "train/" + to_search)
absolute_test_dir = os.path.join(cwd, "test/" + to_search)

try:
    os.makedirs(absolute_train_dir, exist_ok=True)
except OSError:
    print(f"Creation of {absolute_test_dir} directory failed")
try:
    os.makedirs(absolute_test_dir, exist_ok=True)
except OSError:
        print(f"Creation of {absolute_test_dir} directory failed")

# Search for elements
driver.get("https://www.shutterstock.com/search/" + to_search + "?image_type=photo") # Only for real photos
try:
    TOTAL_PAGES = driver.find_element_by_class_name("b_ay_g").text # Obtain number of pages
    TOTAL_PAGES = int(TOTAL_PAGES.split()[1].replace(",",""))
    MAX_IMAGES = driver.find_element_by_xpath("//small/h2") # Total number of images
    MAX_IMAGES = int(MAX_IMAGES.text.split()[0].replace(",", ""))
    NEXT_button = driver.find_element_by_xpath("//div[@class='z_b_g']/a")

    target_images = int(sys.argv[2])
    downloaded_images = 0
    visited_pages = 0
    imgs_url = []

    if MAX_IMAGES < target_images:
        target_images = MAX_IMAGES
        print(f"Max number of images reached: {MAX_IMAGES}")

    # Get url's
    while len(imgs_url) < target_images:
        container = driver.find_elements_by_css_selector("img.z_h_a.z_h_b")

        for img in container:
            if len(imgs_url) < target_images:
                try:
                    imgs_url.append(img.get_attribute("src"))
                except:
                    pass
            
        if visited_pages < TOTAL_PAGES:
            NEXT_button.click()
            visited_pages += 1

    # Download the images
    os.chdir(absolute_train_dir)
    for to_download in range(int(len(imgs_url)*0.8)):
        downloadImage(imgs_url[to_download], "train" + str(to_download + 1) + ".jpg")
            
    os.chdir(absolute_test_dir)
    for to_download in range(int(len(imgs_url)*0.8), int(len(imgs_url))):
        downloadImage(imgs_url[to_download], "test" + str(to_download + 1) + ".jpg")
    
except:
    print(f"Failed to download {to_search} images")

os.remove(cwd + "/geckodriver.log")
driver.quit()
