import cv2

#acessando o arquivo haarcascade
detector_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# abrindo a webcam -> 0
video_capture = cv2.VideoCapture(0)

# loop infinito para captura dos frames (video quebrado em frames)
while True:
    # Captura frame por frame
    ok, frame = video_capture.read()
    #criando imagem cinza convertendo o frame para cinza
    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #fazendo a detaccao na imagm cinza
    deteccoes = detector_face.detectMultiScale(imagem_cinza)
    
    # Desenha retangulo
    for (x, y, w, h) in deteccoes:
        print(w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostra o retangulo no vídeo
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberando a memória no final
video_capture.release()
cv2.destroyAllWindows()