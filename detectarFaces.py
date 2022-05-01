import cv2
import os
import time

cont = 0
naoTemFace = 0
tempos = []

# Tempo inicial
tempInicial = time.time()

# Acessar o diretorio Banco Imagens
caminhos = [os.path.join('Banco Imagens', f) for f in os.listdir('Banco Imagens')]

# Verificar se cada arquivo tem uma face, se não tiver contar mais 1
for caminhoImagem in caminhos:
    # Tempo inicial de cada imagem
    tempInicialImagem = time.time()
    imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
    # Redimensionando a imagem para 400x400
    imagemFace = cv2.resize(imagemFace, (400, 400))
    detectorFace = cv2.CascadeClassifier('C:\\Users\\gog_e\\OneDrive\\Ambiente de Trabalho\\Eigenfaces v2\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml')
    detectorEyes = cv2.CascadeClassifier('C:\\Users\\gog_e\\OneDrive\\Ambiente de Trabalho\\Eigenfaces v2\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')
    detectorGlass = cv2.CascadeClassifier('C:\\Users\\gog_e\\OneDrive\\Ambiente de Trabalho\\Eigenfaces v2\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_eye_tree_eyeglasses.xml')
    facesDetectadas = detectorFace.detectMultiScale(imagemFace, scaleFactor=1.1, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)
     # Criar a bouding box na imagem
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagemFace, (x, y), (x + l, y + a), (0, 0, 255), 2)
        # Criar um bounding box nos olhos
        olhosDetectados = detectorEyes.detectMultiScale(imagemFace, scaleFactor=1.1, minNeighbors=8, flags=cv2.CASCADE_SCALE_IMAGE)
        
    if len(facesDetectadas) == 0:
        naoTemFace += 1
        # Move a imagem para o diretorio Sem Faces
        os.rename(caminhoImagem, 'Sem Faces\\' + str(naoTemFace) +'-semFace' + '.jpg')

    else:
        cont += 1
        # Acessa o diretório Sem Faces e move o arquivo para o diretório Sem Faces
        os.rename(caminhoImagem, 'Faces Encontradas\\' + str(cont) + '-treinamento' + str(cont) + '.jpg')

    # Tempo final de cada imagem
    tempFinalImagem = time.time()
    # Adiciona o tempo de cada imagem ao vetor de tempos
    tempos.append(tempFinalImagem - tempInicialImagem)

# Apaga o diretório Banco Imagens
os.rmdir('Banco Imagens')

# Tempo final
tempFinal = time.time()

# Mostra o tempo total de execução
print('Tempo total de execução: ', (tempFinal - tempInicial) / 1000, 's')
# Mostra o tempo médio de execução
print('Tempo médio de execução: ', ((tempFinal - tempInicial)/cont) / 1000, 's')
# Mostra o tempo médio de execução
print('Tempo médio de execução: ', sum(tempos)/len(tempos))
# Mostrar o maior tempo de execução
print('Maior tempo de execução: ', max(tempos))
# Mostrar o menor tempo de execução
print('Menor tempo de execução: ', min(tempos))
# Mostrar quantidade total de imagens
print('Quantidade total de imagens: ', len(caminhos))
# Mostrar quantidade de imagens sem faces
print('Quantidade de imagens sem faces: ', naoTemFace)
# Mostrar quantidade total de imagens com face
print('Quantidade total de imagens com face: ', cont)

