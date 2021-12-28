import cv2
import numpy as np
import os
from PIL import Image

path = os.getcwd()
print('__file__:     ', __file__)
print(path)
pathFile = os.path.dirname(os.path.abspath(__file__))
print(pathFile)
os.chdir(pathFile)

pathFileFotos = pathFile + '\\imagens\\treinamento'
print(pathFileFotos)

eigenface = cv2.face.EigenFaceRecognizer_create(40, 8000)

def pegarImagemComID():

    caminhos = [os.path.join(pathFileFotos, f) for f in os.listdir(pathFileFotos)]
    print(caminhos)

    faces = []
    ids = []

    for caminhoImagem in caminhos:
        imagemFace = Image.open(caminhoImagem).convert('L')
        imagemNP = np.array(imagemFace, 'uint8')
        id = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject",""))
        ids.append(id)
        faces.append(imagemNP)

    return np.array(ids), faces

ids, faces = pegarImagemComID()

print(ids)
print(faces)

eigenface.train(faces, ids)
eigenface.write(pathFile + '\\classificadorEigen.yml')

print('\n\nTreinamento feito com sucesso !!!')







