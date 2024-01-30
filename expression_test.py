import cv2
import dlib
import numpy as np
# Load pre-trained facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Download from dlib's website
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Read two images
image1 = cv2.imread("lord_ram1.png")
image2 = cv2.imread("lord_ram2.png")

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect faces in the images
faces1 = detector(gray1)
faces2 = detector(gray2)

# Loop through the faces and draw all 68 facial landmarks
for face in faces1:
    landmarks = predictor(gray1, face)

    for i in range(68):
        # Draw circle for each facial landmark
        cv2.circle(image1, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
for face in faces2:
    landmarks = predictor(gray2, face)

    for i in range(68):
        # Draw circle for each facial landmark
        cv2.circle(image2, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
# Loop through the faces and compare landmarks for eyes and lips
for face1, face2 in zip(faces1, faces2):
    landmarks1 = predictor(gray1, face1)
    landmarks2 = predictor(gray2, face2)

    # Extract coordinates of specific facial landmarks for eyes
    left_eye1 = [(landmarks1.part(36).x, landmarks1.part(36).y),
                 (landmarks1.part(39).x, landmarks1.part(39).y)]
    right_eye1 = [(landmarks1.part(42).x, landmarks1.part(42).y),
                  (landmarks1.part(45).x, landmarks1.part(45).y)]

    left_eye2 = [(landmarks2.part(36).x, landmarks2.part(36).y),
                 (landmarks2.part(39).x, landmarks2.part(39).y)]
    right_eye2 = [(landmarks2.part(42).x, landmarks2.part(42).y),
                  (landmarks2.part(45).x, landmarks2.part(45).y)]

    # Extract coordinates of specific facial landmarks for lips
    outer_lip1 = [(landmarks1.part(48).x, landmarks1.part(48).y),
                  (landmarks1.part(54).x, landmarks1.part(54).y)]
    outer_lip2 = [(landmarks2.part(48).x, landmarks2.part(48).y),
                  (landmarks2.part(54).x, landmarks2.part(54).y)]

    inner_lip1 = [(landmarks1.part(60).x, landmarks1.part(60).y),
                  (landmarks1.part(64).x, landmarks1.part(64).y)]
    inner_lip2 = [(landmarks2.part(60).x, landmarks2.part(60).y),
                  (landmarks2.part(64).x, landmarks2.part(64).y)]

    # Compare the coordinates and highlight changes for eyes
    for (pt1, pt2) in zip(left_eye1 + right_eye1, left_eye2 + right_eye2):
        cv2.circle(image2, pt2, 2, (0, 0, 255), -1)

    # Compare the coordinates and highlight changes for lips
    for (pt1, pt2) in zip(outer_lip1 + inner_lip1, outer_lip2 + inner_lip2):
        cv2.circle(image2, pt2, 2, (0, 0, 255), -1)
height = 387
image1_resized = cv2.resize(image1, (int(height * image1.shape[1] / image1.shape[0]), height))
image2_resized = cv2.resize(image2, (int(height * image2.shape[1] / image2.shape[0]), height))

# Concatenate images side by side
vis = np.concatenate((image1_resized, image2_resized), axis=1)

cv2.imshow('Facial Difference', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()