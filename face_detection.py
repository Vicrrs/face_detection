import cv2 as cv

img = cv.imread('pessoas.jpg')

classificador = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

detecta = classificador.detectMultiScale(gray)

print(detecta)
print(len(detecta))

for (x, y, l, a) in detecta:
    cv.rectangle(img, (x, y), (x+l, y+a), (0, 255, 0), 2)

cv.imshow('Detector de faces', img)
cv.waitKey(0)
cv.destroyAllWindows()
