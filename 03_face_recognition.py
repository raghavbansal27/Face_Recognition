import cv2

import tkinter as tk
window = tk.Tk()
window.title("Face Detection")
def main():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(r'C:\Users\ragha\PycharmProjects\FaceDetection\trainer\trainer.yml')
    # Create the CascadeClassifier object
    face_cascade = cv2.CascadeClassifier(r"D:\data_sets\haarcascade_frontalface_default.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX

    id = 0
    names = ['None', 'Raghav', 'Dolly', 'Keshav', 'Deepak']
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        check, frame = video.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for x, y, w, h in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        cv2.imshow("Video", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    exit_label = tk.Label(window, text='[INFO] Exiting Program and cleanup stuff')
    exit_label.grid(column=0, row=6)
    video.release()
    cv2.destroyAllWindows()

window.geometry('600x600')

label = tk.Label(window, text="Face Recognition")
label.grid(column=0, row=1)

button = tk.Button(window, text="Press to open Camera", command=main)
button.grid(column=0, row=2)

exit = tk.Label(window, text="Press 'q' to exit camera")
exit.grid(column=0, row=5)
window.mainloop()