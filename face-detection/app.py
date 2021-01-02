import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")

# img = cv2.imread(
#     'group.jpeg')

webcam = cv2.VideoCapture(0)

while True:
    successfull_frame_read, frame = webcam.read()
    print(frame)
    grey_scaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grey_scaled_image)
    for face_coordinate in face_coordinates:
        (x, y, w, h) = face_coordinate
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256),
                                                  randrange(256), randrange(256)), 2)
    cv2.imshow("Detected face", frame)
    cv2.waitKey(10)


key = cv2.waitKey()

print(str(key)+"pressed")


# grey_scaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face_coordinates = trained_face_data.detectMultiScale(grey_scaled_image)

# print(face_coordinates)

# for face_coordinate in face_coordinates:
#     (x, y, w, h) = face_coordinate
#     cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),
#                                             randrange(256), randrange(256)), 2)


# cv2.imshow("Detected face", img)
# cv2.waitKey()
# print("Code of face detection completed")
