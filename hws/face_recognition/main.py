import pathlib
import pickle

import cv2
import face_recognition
import numpy as np


KNOWN_PERSONS_PATH = 'persons'
ME_FILENAME = 'me.jpg'
ME_ENCODING_FILENAME = "me.dat"


def save_face_enc(face_enc: np.ndarray) -> None:
    print(type(face_enc))
    with open(ME_ENCODING_FILENAME, 'wb') as f:
        pickle.dump(face_enc, f)


def retrieve_face_enc() -> np.ndarray:
    with open(ME_ENCODING_FILENAME, 'rb') as f:
        me_enc = pickle.load(f)
    return me_enc


def get_my_face_enc() -> np.ndarray:
    if pathlib.Path(ME_ENCODING_FILENAME).exists():
        return retrieve_face_enc()
    me = face_recognition.load_image_file(
        str(pathlib.PurePath(KNOWN_PERSONS_PATH, ME_FILENAME))
    )
    me_encoding = face_recognition.face_encodings(me)[0]
    save_face_enc(me_encoding)
    return me_encoding


def run():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Cannot open camera")
        return

    me_encoding = get_my_face_enc()
    face_locations = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if process_this_frame:
            # Resize frame of video to 1/4 size
            # for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses)
            # to RGB color (which face_recognition uses)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            face_locations = face_recognition.face_locations(
                rgb_small_frame
            )
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces([me_encoding], face_encoding)
                name = 'Unknown'
                if True in matches:
                    name = 'Lena Gordeeva'
                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
            )

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
