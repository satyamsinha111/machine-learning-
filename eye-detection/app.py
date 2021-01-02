import cv2
trained_eye_data = cv2.CascadeClassifier("haarcascade_eye.xml")

webcam = cv2.VideoCapture(0)

while True:
    success_frame_read, frame = webcam.read()
    grey_scale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_coordinates = trained_eye_data.detectMultiScale(grey_scale_image)
    print(eye_coordinates)
    for eye_coordinate in eye_coordinates:
        (x, y, w, h) = eye_coordinate
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow("Grey scale image", frame)
    cv2.waitKey(1)


# img = cv2.imread("eyes.jpg")
# grey_scale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# eye_coordinates = trained_eye_data.detectMultiScale(grey_scale_image)
# print(eye_coordinates)
# for eye_coordinate in eye_coordinates:
#     (x, y, w, h) = eye_coordinate
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
# cv2.imshow("Grey scale image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
