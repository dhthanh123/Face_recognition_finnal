import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import tkinter as tk
from tkinter import messagebox


def collate_fn(x):
    return x[0]



if __name__ == "__main__":

    device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("the code running on: ",device)
    IMG_PATH = './photos/'
    mtcnn0 = MTCNN(image_size=160, margin=0, keep_all=False, min_face_size=40, device=device)  # keep_all=False
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, min_face_size=40, device=device)  # keep_all=True
    resnet = InceptionResnetV1(pretrained='vggface2').to(device)
    resnet.eval()

    #embedding_list, name_list = loaddata(data_path)
    dataset = datasets.ImageFolder(IMG_PATH)
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
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
    torch.save(data, './Data/data.pt') # saving data.pt file
    messagebox.showinfo(title="Thông Báo", message="Cập nhật dữ liệu hoàn tất.")




