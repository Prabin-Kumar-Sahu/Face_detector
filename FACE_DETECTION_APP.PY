import cv2
from random import randrange
# print(cv2.__version__)
# print("code")

# load some  pre-trained data on face frontals from opencv (harr casecade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# choose an image to detect faces in
# img = cv2.imread('fce-header-motif-7.png')

# to Capture video from Webcam
webcam = cv2.VideoCapture(0)
key = cv2.waitKey(1)
# must convert to grayscale
# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)

# Draw rectangles around the faces for single face
# (x, y, w, h) = face_coordinates[0]
# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)  # or
# cv2.rectangle(img, (116, 50), (236, 170), (0, 255, 0), 2)

# Draw rectangles around the faces for multiple face
for(x, y, w, h) in face_coordinates:
    # (x, y, w, h) = face_coordinates[0]
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),
                  randrange(256), randrange(256)), 5)


# cv2.imshow("Prabin Face Detector App", grayscaled_img)
# cv2.waitKey()  # waitkey use for to stop the program to terminate immediately

# cv2.imshow("Prabin Face Detector App", img)  # it shows the color photo
# cv2.waitKey()  # waitkey use for to stop the program to terminate immediately
