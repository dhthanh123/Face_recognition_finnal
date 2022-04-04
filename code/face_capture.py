import sys

import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from datetime import datetime
import os
import tkinter as tk
from tkinter import messagebox
import unidecode
import  hashlib
import shutil

IMG_PATH = './stamp/'
Photo_dic = './photos/'
root = tk.Tk()
name_var = tk.StringVar()
passw_var = tk.StringVar()

def load_data(data_path):
    data = torch.load(data_path)
    embedding_list = data[0]
    name_list = data[1]
    return name_list, embedding_list

def inarr(string, arr):
    for e in arr:
        if string == e:
            return True
    return False

def save_data(embedding_list, name_list, data_path):
    data = [embedding_list, name_list]
    torch.save(data, data_path+'data.pt')# saving data.pt file

def collate_fn(x):
    return x[0]

def remove_accent(text):
    return unidecode.unidecode(text)

def md5_encode(string):
    hash = string.strip().encode("utf-8")
    hash = hashlib.md5((hash))
    return hash.hexdigest()


def add_staff():
    name = remove_accent(name_var.get())
    name=name.upper()
    password = remove_accent(passw_var.get())
    password = md5_encode(password)
    if name!='' and password !='':
        try:
            count = 10
            staff_code = name
            staff_name = password
            name = staff_code + '_' + staff_name
            USR_PATH = os.path.join(IMG_PATH, name)
            leap = 1
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print("code run on: ", device)
            mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, post_process=False, device=device)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            while cap.isOpened() and count:
                isSuccess, frame = cap.read()
                if mtcnn(frame) is not None and leap % 2:
                    path = str(USR_PATH + '/{}.jpg'.format(
                        str(datetime.now())[:-7].replace(":", "-").replace(" ", "-") + str(count)))
                    face_img = mtcnn(frame, save_path=path)
                    count -= 1
                leap += 1
                cv2.imshow('Thu Du Lieu....', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()

            data_path = './Data/data.pt'
            photo_dir = './stamp/'
            dataset = datasets.ImageFolder(photo_dir)
            idx_to_class = {i: c for c, i in
                            dataset.class_to_idx.items()}  # accessing names of peoples from folder names
            name_list, embedding_list = load_data(data_path)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            if inarr(name, name_list) == False:
                # initializing MTCNN and InceptionResnetV1
                mtcnn0 = MTCNN(image_size=160, margin=0, keep_all=False, min_face_size=40,
                               device=device)  # keep_all=False
                resnet = InceptionResnetV1(pretrained='vggface2').to(device)
                resnet.eval()
                loader = DataLoader(dataset, collate_fn=collate_fn)

                for img, idx in loader:
                    face, prob = mtcnn0(img, return_prob=True)
                    if face is not None and prob > 0.92:
                        emb = resnet(face.unsqueeze(0).to(device))
                        embedding_list.append(emb.detach())
                        name_list.append(idx_to_class[idx])  # name of folder
                # print(name_list)
                save_data(embedding_list, name_list, './Data/')
                shutil.move(photo_dir + name, Photo_dic)
                messagebox.showinfo(title="Thông Báo",
                                    message="Thêm thông tin nhân viên thành công! reload lại chương trình để kiểm tra!")
            else:
                messagebox.showinfo(title="Thông Báo",message="Nhân Viên này đã được thêm vào hệ thống trước đó. Vui lòng kiểm tra lại:::!")
        except:
            messagebox.showinfo(title="Thông Báo",message="Lỗi!!! Vui lòng liên hệ bộ phận kỹ thuật:::!")
            sys.exit()
        root.destroy()

    else:
        messagebox.showinfo(title="Thông Báo", message="Vui Lòng Nhập đầy đủ mã và tên nhân viên:")

root.geometry("400x100")

name_label = tk.Label(root, text='Mã nhân viên', font=('UTM Avo', 10, 'bold'))
name_entry = tk.Entry(root, textvariable=name_var, font=('UTM Avo', 10, 'normal'))
name_entry.config(width=60,font=("Courier", 34))
passw_label = tk.Label(root, text='Mật khẩu', font=('UTM Avo', 10, 'bold'))
passw_entry = tk.Entry(root, textvariable=passw_var, font=('UTM Avo', 10, 'normal'), show="*")
passw_entry.config(width=60,font=("Courier", 34))
sub_btn = tk.Button(root, text='Thêm nhân viên', command=add_staff, font=('UTM Avo', 10, 'normal'))
name_label.grid(row=0, column=0)
name_entry.grid(row=0, column=1)
passw_label.grid(row=1, column=0)
passw_entry.grid(row=1, column=1)
sub_btn.grid(row=2, column=1)
root.mainloop()
