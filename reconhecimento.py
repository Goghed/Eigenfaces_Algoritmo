import cv2
import os
import numpy as np
import time

cont = 0
idReconhecido = 0
tempos = []

reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read('classificadorEigen.yml')

# Acessar o diretorio Imagens Teste
caminhos = [os.path.join('Imagens Teste', f) for f in os.listdir('Imagens Teste')]

# Tempo inicial da verificação
tempInicial = time.time()

# Detectar se há uma face nas imagens do diretorio e marcar com um bounding box
for caminhoImagem in caminhos:
    # Tempo inicial de cada imagem
    tempInicialImagem = time.time()
    imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
    imagemFace = cv2.resize(imagemFace, (400, 400))
    # Detectar a face na imagem
    faceDetect = cv2.CascadeClassifier('C:\\Users\\gog_e\\OneDrive\\Ambiente de Trabalho\\Eigenfaces v2\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    faces = faceDetect.detectMultiScale(imagemFace, scaleFactor=1.1,  minNeighbors=4, minSize=(200, 200))
    # Criar a bounding box
    for (x, y, l, a) in faces:
        imagemFace = cv2.rectangle(imagemFace, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagemFace)        
        print('ID da imagem que contém as mesmas caracteristicas no diretorio de faces reconhecidas: {}'.format(id))        
        print('Confiança: {}'.format(confianca))
        idReconhecido += 1
        cv2.imshow('Face de Teste', imagemFace)
        cv2.waitKey(0)
    
    # Tempo final de cada imagem
    tempFinalImagem = time.time()
    # adicionar na lista de tempos
    tempos.append(tempFinalImagem - tempInicialImagem)

    # Pegar a imagem do diretório Banco Imagens que contém o mesmo ID do predict e mostrar na tela
    caminhos = [os.path.join('Faces Encontradas', f) for f in os.listdir('Faces Encontradas')]
    for caminhoImagem in caminhos:
        if int(os.path.split(caminhoImagem)[-1].split('-')[0]) == id:
            imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
            imagemFace = cv2.resize(imagemFace, (400, 400))
            cont += 1
            cv2.imshow('Face do Banco Imagens', imagemFace)
            cv2.waitKey(0)

# Tempo final da verificação
tempFinal = time.time()

# Acessar o diretorio Imagens Teste
caminhos = [os.path.join('Imagens Teste', f) for f in os.listdir('Imagens Teste')]
# Contar a quantidade de arquivos
print('Quantidade de imagens no diretório de teste: {}'.format(len(caminhos)))
print('Quantidade de imagens que foram reconhecidas: {}'.format(idReconhecido))
print('Quantidade de imagens que não foram reconhecidas: {}'.format(cont - idReconhecido))
# taxa de porcentagem de acerto mostrando até dois algorismos apos o .99
print('Taxa de acerto: {:.2f}%'.format((idReconhecido / len(caminhos)) * 100))
# Mostrar o threshold de confiança
print('Acuracia da predict: {}'.format(reconhecedor.getThreshold()))
# tempo total do reconhecimento
print('Tempo total de reconhecimento: {:.2f} segundos'.format(tempFinal - tempInicial))
# tempo medio de cada imagem
print('Tempo medio de reconhecimento de cada imagem: {:.2f} segundos'.format(sum(tempos) / len(tempos)))
# maior tempo de reconhecimento
print('Maior tempo de reconhecimento: {:.2f} segundos'.format(max(tempos)))
# menor tempo de reconhecimento
print('Menor tempo de reconhecimento: {:.2f} segundos'.format(min(tempos)))




