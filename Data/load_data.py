# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import datetime
import os
import numpy as np
import requests
import json
import unidecode
import http.client as httplib

def load_data(data_path):
    data = torch.load(data_path)
    embedding_list = data[0]
    name_list = data[1]
    return name_list, embedding_list

def save_data(embedding_list, name_list, data_path):
    data = [embedding_list, name_list]
    torch.save(data, data_path+'data.pt')# saving data.pt file
def general_data(dataset,idx_to_class):
    loader = DataLoader(dataset, collate_fn=collate_fn)
    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

    for img, idx in loader:
        face, prob = mtcnn0(img, return_prob=True)
        if face is not None and prob>0.92:
            emb = resnet(face.unsqueeze(0).to(device))
            embedding_list.append(emb.detach())
            name_list.append(idx_to_class[idx])

    # save data
    data = [embedding_list, name_list]
    torch.save(data, 'data.pt') # saving data.pt file

def collate_fn(x):
    return x[0]

if __name__=="__main__":
    data_path = '../data.pt'
    photo_dir = '../new/'
    load_data(data_path)
    dataset = datasets.ImageFolder(photo_dir)
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}  # accessing names of peoples from folder names
    embedding_list, name_list = load_data(data_path)  # load data to list
    name_list,embedding_list = load_data(data_path)
# check which device being used
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("code run on: ", device)
# initializing MTCNN and InceptionResnetV1
    mtcnn0 = MTCNN(image_size=160, margin=0, keep_all=False, min_face_size=40, device=device)  # keep_all=False
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, min_face_size=40, device=device)  # keep_all=True
    resnet = InceptionResnetV1(pretrained='vggface2').to(device)
    resnet.eval()

    loader = DataLoader(dataset, collate_fn=collate_fn)

    for img, idx in loader:
        face, prob = mtcnn0(img, return_prob=True)
        if face is not None and prob > 0.92:
            emb = resnet(face.unsqueeze(0).to(device))
            embedding_list.append(emb.detach())
            name_list.append(idx_to_class[idx])  # name of folder

    print(name_list)

    save_data(embedding_list,name_list, '../new/data.pt')



