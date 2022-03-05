import cv2
from random import randrange


# load some  pre-trained data on face frontals from opencv (harr casecade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


# to Capture video from Webcam
webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture(
#     "Ghana_Pallbearers_Dancing_to_Astronomia_2k19.mp4")
# itrerate forever over frames
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces for multiple face
    for(x, y, w, h) in face_coordinates:
        # (x, y, w, h) = face_coordinates[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256),
                                                  randrange(256), randrange(256)), 5)
    cv2.imshow('PRABIN FACE DETECTOR APP', frame)
    key = cv2.waitKey(1)

    # stop if Q key is pressed
    if key == 81 or key == 113:
        break

    # Release the videoCapture object
    # webcam.release()
