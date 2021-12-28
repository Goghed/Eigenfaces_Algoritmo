import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score

path = os.getcwd()
print('__file__:     ', __file__)
print(path)
pathFile = os.path.dirname(os.path.abspath(__file__))
print(pathFile)
os.chdir(pathFile)

pathFileFotos = pathFile + '\\imagens\\teste'
print(pathFileFotos)

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

faces = np.asarray(faces)

print(ids)
print(faces)

reconhecedorEigen = cv2.face.EigenFaceRecognizer_create()
reconhecedorEigen.read(pathFile + '\\classificadorEigen.yml')
id_predict_Eigen = []
confiancaEigen_P = []

for indice in range(len(ids)):
    idEigen, confiancaEigen = reconhecedorEigen.predict(faces[indice])
    id_predict_Eigen.append(idEigen)
    confiancaEigen_P.append(confiancaEigen)
acuraciaEigen = accuracy_score(ids, id_predict_Eigen)

resultados = pd.DataFrame()
resultados['ids'] = ids
resultados['id_Eigen'] = id_predict_Eigen
resultados['confianca_Eigen'] = confiancaEigen_P
resultados.to_csv(pathFile + '\\Tabela_resultados.csv', sep = ';')

print('\n')

print(resultados)

print('>>>>>> Acuracia <<<<<<')
print('Eigen: ', acuraciaEigen)


