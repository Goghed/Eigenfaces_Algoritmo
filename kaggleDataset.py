import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import cv2
from PIL import Image

api = KaggleApi()
api.authenticate()

# Criar o diretorio dataset_training
if not os.path.exists('Banco Imagens'):
    os.makedirs('Banco Imagens')

# Baixar o dataset de Yale Faces e salvar em dataset_training
kaggle.api.dataset_download_files('asacxyz/ic-fatecitu', path='Banco Imagens', unzip=True)

cont = 1
naoTemFace = 0
tempos = []

# Acessar o diretorio Banco Imagens
caminhos = [os.path.join('Banco Imagens', f) for f in os.listdir('Banco Imagens')]

# Remover a extensão de todos os arquivos do diretorio
for caminhoImagem in caminhos:
    
    if ".jpeg" in caminhoImagem:                
        # pega o nome do arquivo em .jpeg e retira a extensão.
        nomeArquivo = os.path.splitext(caminhoImagem)[0]
        # converte o arquivo .jpeg para .jpg ja passando o nome inicial do arquivo e adicionando a extensão .jpg
        imagem = Image.open(caminhoImagem).convert('RGB').save(nomeArquivo+'.'+'jpg')        
        # remove o arquivo .jpeg
        os.remove(caminhoImagem)
        

    if ".jpg" in caminhoImagem:
        # Le a imagem
        imagem = cv2.imread(caminhoImagem)
        if imagem is None:
            # Deleta a imagem caso não tenha sido possivel ler
            os.remove(caminhoImagem)
            naoTemFace += 1
        # Converte para escala de cinza
        img = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        # Redimensiona a imagem para 256x256
        img = cv2.resize(img, (600, 600))
        # Salva a imagem
        cv2.imwrite(caminhoImagem, img)        
        # mudar nome do arquivo para {cont}-treinamento{cont}.jpg
        os.rename(caminhoImagem, os.path.join('Banco Imagens', str(cont)+'-treinamento'+str(cont)+'.jpg'))
        cont += 1
    
    
        