# import cv2
# import dlib
# import numpy as np
# from imutils import face_utils
#
# # Load the pre-trained models
# face_detector = dlib.get_frontal_face_detector()
# landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
# def detect_faces(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     rects = face_detector(gray, 1)
#     return rects
#
# def draw_landmarks(image, face_rects):
#     for rect in face_rects:
#         shape = landmark_predictor(image, rect)
#         shape = face_utils.shape_to_np(shape)
#
#         for (x, y) in shape:
#             cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
#
#     return image
#
# def main():
#     cap = cv2.VideoCapture(0)
#
#     while True:
#         ret, frame = cap.read()
#         face_rects = detect_faces(frame)
#         frame = draw_landmarks(frame, face_rects)
#
#         cv2.imshow("Face Recognition", frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     main()

import cv2
import dlib
import numpy as np
from imutils import face_utils
import matplotlib.pyplot as plt

# Load the pre-trained models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 1)
    return rects


def draw_landmarks(image, face_rects):
    for rect in face_rects:
        shape = landmark_predictor(image, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return image


def main():
    # Read the image
    image = cv2.imread("face.jpg")

    # Detect faces and draw landmarks
    face_rects = detect_faces(image)
    image = draw_landmarks(image, face_rects)

    # Show the image with landmarks using matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Face Recognition")
    plt.show()


if __name__ == '__main__':
    main()