import os
import requests
from labelbox import Client
import json
from collections import namedtuple
import random
from math import floor
import cv2

Dataset = namedtuple('Dataset', 'id original mask')


# create folder
def createFolder(path):
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

# create folders
def createFolders(paths):
    for folder in paths:
        createFolder(folder)

# download from url and save
def download(name, url):
    r = requests.get(url)
    with open(name, "wb") as file:
        file.write(r.content)

    return name

# download labelbox json from labelbox api
def download_labelbox_json(token_api, project_id, path):
    labelbox_client = Client(token_api)
    project = labelbox_client.get_project(project_id)
    url = project.export_labels()
    download(path, url)

# filter only images and masks from labelbox json
def filter_labelbox_json(path_json, ):
    labelbox_json = None
    with open(path_json) as f:
        labelbox_json = json.load(f)

    dataset_images = []
    for labelbox in labelbox_json:
        if len(labelbox['Label']) > 0:
            id = labelbox['ID']
            original = labelbox['Labeled Data']
            mask = labelbox['Label']['objects'][0]['instanceURI']

            dataset_images.append(Dataset(id=id, original=original, mask=mask))

    return dataset_images

def to_gray_scale(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(filename, image)

def invert_color(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = cv2.bitwise_not(image)
    cv2.imwrite(filename, image)

def split_image_arr(images, testPercent = 0.3):
    random.shuffle(images)
    test_data = []
    train_data = []

    totalSize = len(images)
    testSize = floor(totalSize * testPercent)

    cont = 0
    while cont < len(images):
        data = images[cont]
        if cont < testSize :
            test_data.append(data)
        else:
            train_data.append(data)

        cont += 1
    
    return train_data, test_data

def gen_test_data(path, images, extension):
    cont = 0
    for image in images:
        test_path = download(os.path.join(path, '{}.{}'.format(cont, extension)), image.original)
        to_gray_scale(test_path)
        cont += 1

def gen_train_data(path_train, path_mask, images, extension):
    cont = 0
    for image in images:
        
        original_path = download(os.path.join(path_train, '{}.{}'.format(cont, extension)), image.original)
        to_gray_scale(original_path)

        mask_path = download(os.path.join(path_mask, '{}.{}'.format(cont, extension)), image.mask)
        
        # labelbox return an image with background black and mask white, 
        # but for unet we need background white and mask black
        invert_color(mask_path)
        
        cont += 1