import cv2
import os
import numpy as np
import time 
import timeit

tempos = []

eigenface = cv2.face.EigenFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('Faces Encontradas', f) for f in os.listdir('Faces Encontradas')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        # Redimensionando a imagem para 400x400
        imagemFace = cv2.resize(imagemFace, (400, 400))
        id = int(os.path.split(caminhoImagem)[-1].split('-')[0])        
        ids.append(id)        
        faces.append(imagemFace)        
        
    return np.array(ids), faces

ids, faces = getImagemComId()

print('Treinando...')

# tempo inicial de treinamento
tempTrainInicial = time.time()

eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

# tempo final de treinamento
tempTrainFinal = time.time()

print('Treinamento realizado')

# mostrar o tempo total de execução do treinamento
print('Tempo total de treinamento: {:.2f} s'.format((tempTrainFinal - tempTrainInicial)/60))
# mostrar o tempo médio de treinamento
print('Tempo médio de treinamento: {:.3f} ms'.format((tempTrainFinal - tempTrainInicial)/len(ids)))
# mostrar quantidade de imagens treinadas
print('Quantidade de imagens treinadas: ', len(ids))




