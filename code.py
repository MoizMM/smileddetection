import cv2

# load some pretrained data
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

smile_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')

# to capture video from webcam
webcam = cv2.VideoCapture(0)
cv2.namedWindow("Window")

while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()

    # if there's an error, abort
    if not successful_frame_read:
        break

    # must convert to grayscale every frame
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_detector.detectMultiScale(grayscaled_img)

    # draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        # get the subframe (using numpy-N dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]

        # must convert to grayscale every frame
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # detect smiles in the face subframe
        smiles = smile_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # draw rectangles around smiles
        for (x_, y_, w_, h_) in smiles:
            cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (50, 50, 200), 4)

        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=3, color=(255, 255, 255))

    cv2.imshow('Smile Detector', frame)
    key = cv2.waitKey(1)

    # press 'q' to exit
    if key == ord('q'):
        break

# release video capture
webcam.release()
cv2.destroyAllWindows()
