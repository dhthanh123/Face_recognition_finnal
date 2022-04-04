import glob

import cv2
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QMessageBox
import addstaff
import sys
from facenet_pytorch import MTCNN, InceptionResnetV1
import unidecode
import hashlib
from PIL import Image

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from datetime import datetime
import os
import shutil


class MainWindow(QMainWindow):
    def __init__(self, IMG_PATH, Photo_dic, photo_dir, data_path):
        super().__init__()
        self.setWindowTitle("Thêm Nhân Viên - Hệ Thống Chấm Công")
        self.uic = addstaff.Ui_addstaff()
        self.uic.setupUi(self)
        self.setFixedSize(341, 164)

        self.IMG_PATH = IMG_PATH
        self.Photo_dic = Photo_dic
        self.photo_dir = photo_dir
        self.data_path = data_path

        MainWindow = QtWidgets.QMainWindow()
        self.uic.btn_submit.clicked.connect(lambda: self.addstaff())
        MainWindow.show()

    def get_from_form(self):
        return self.uic.staff_code.text(), self.uic.staff_pass.text()

    def load_data(self):
        data = torch.load(self.data_path)
        embedding_list = data[0]
        name_list = data[1]
        return name_list, embedding_list


    def inarr(self, string, arr):
        for e in arr:
            if string == e:
                return True
        return False

    def save_data(self, embedding_list, name_list):
        data = [embedding_list, name_list]
        torch.save(data, self.data_path)# saving data.pt file

    def collate_fn(self, x):
        return x[0]

    def remove_accent(self, text):
        return unidecode.unidecode(text)

    def md5_encode(self, string):
        hash = string.strip().encode("utf-8")
        hash = hashlib.md5((hash))
        return hash.hexdigest()

    def load_img(self, path):
        list_img = []
        for filename in glob.glob(path + "/*.jpg"):
            img_load = Image.open(filename)
            list_img.append(img_load)
        return list_img

    def addstaff(self):
       staff_code, staff_pass = self.get_from_form()

       if staff_code.strip() != '' and staff_pass.strip() != '':
            name = staff_code + '_' + self.md5_encode(staff_pass)
            name_list, embedding_list = self.load_data()
            USR_PATH = os.path.join(self.IMG_PATH, name)
            if os.path.isdir(USR_PATH) == False:
                os.makedirs(USR_PATH)



            if self.inarr(name, name_list) == False:
                count = 10
                leap = 1
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                print("code run on: ", device)

                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                while cap.isOpened() and count:
                    isSuccess, frame = cap.read()  # and count:
                    if mtcnn(frame) is not None and leap % 2:
                        path = str(USR_PATH + '/{}.jpg'.format(str(datetime.now())[:-7].replace(":", "-").replace(" ", "-") + str(count)))
                        face_img = mtcnn(frame, save_path=path)
                        count -= 1
                    leap += 1
                    cv2.imshow('Thu Du Lieu....', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                cap.release()
                cv2.destroyAllWindows()


                dataset = datasets.ImageFolder(self.photo_dir)
                loader = DataLoader(dataset, collate_fn=self.collate_fn)
                idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

                for img, idx in loader:
                    face, prob = mtcnn(img, return_prob=True)
                    if face is not None and prob > 0.92:
                        emb = resnet(face.unsqueeze(0).to(device))
                        embedding_list.append(emb.detach())
                        name_list.append(idx_to_class[idx])

                self.save_data(embedding_list, name_list)
                shutil.move(photo_dir + name, self.Photo_dic)
                QMessageBox.about(self, 'Thông báo',"Thêm thông tin nhân viên thành công! reload lại chương trình để kiểm tra!")
                sys.exit()
            else:
                 QMessageBox.about(self, 'Thông báo',"Nhân Viên này đã được thêm vào hệ thống trước đó. Vui lòng kiểm tra lại:::!")

       else:
           QMessageBox.about(self, 'Thông báo', "Vui Lòng Nhập Mã Nhân Viên và Mật Khẩu")



if __name__ == "__main__":

    IMG_PATH = './stamp/'
    Photo_dic = './photos/'
    photo_dir = './stamp/'
    data_path = './Data/data.pt'

    app = QApplication(sys.argv)
    main_win = MainWindow(IMG_PATH, Photo_dic,photo_dir, data_path)
    main_win.show()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, min_face_size=40,device=device)  # keep_all=False
    resnet = InceptionResnetV1(pretrained='vggface2').to(device)
    resnet.eval()
    sys.exit(app.exec())