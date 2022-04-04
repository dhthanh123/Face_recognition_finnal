import torch.onnx
from facenet_pytorch import MTCNN, InceptionResnetV1

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=128, margin=0, keep_all=True, min_face_size=40, device=device)
    dummy_input = torch.randn(1, 3, 160, 160, device=device)
    model = InceptionResnetV1(pretrained='vggface2').to(device)
    model.eval()


    input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
    output_names = ["output1"]

    torch.onnx.export(model, dummy_input, "facenet.onnx", verbose=True, input_names=input_names, output_names=output_names)
