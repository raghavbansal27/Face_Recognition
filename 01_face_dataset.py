import cv2

# Create the CascadeClassifier object
face_cascade = cv2.CascadeClassifier(r"D:\data_sets\haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_id = input('\n Enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

count = 0
while True:
    check, frame = video.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("C:\\Users\\ragha\\PycharmProjects\\FaceDetection\\dataset\\User." + str(face_id) + '.' + str(count) + ".jpg", frame[y:y + h, x:x + w])
        cv2.imshow("Video", frame)
    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    elif count >= 30:  # Take 30 face sample and stop video
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
video.release()
cv2.destroyAllWindows()