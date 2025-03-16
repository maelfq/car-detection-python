import os

import cv2
import imutils
import numpy as np
from cv2 import Mat
from numpy import ndarray
import easyocr


def detect_licence_plate(image: Mat | ndarray) -> str:
    licence_plate_image = get_licence_plate_cropped(image)
    texts = read_text_from_image(licence_plate_image)
    print(texts)
    if len(texts) >= 1:
        save_to_database(texts[0])
        return texts[0]
    return ""


def read_text_from_image(image: Mat | ndarray) -> list[str]:
    tmp_image_path: str = os.getcwd() + os.sep + 'tmp.jpeg'
    cv2.imwrite(tmp_image_path, image)
    reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
    result = reader.readtext(tmp_image_path)
    print(result)
    os.remove(tmp_image_path)

    predicted_license_plates = []

    for detection in result:
        print(detection)
        predicted_license_plates.append(detection[1])

    return predicted_license_plates


def save_to_database(licence_plate: str) -> bool:
    # TODO
    return False


def get_licence_plate_cropped(image: Mat | ndarray) -> Mat | ndarray:
    # code to get licence plate found here:
    # https://cyberworrier2000.medium.com/license-plate-recognition-using-opencv-python-e03dd591f083

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)

    edged = cv2.Canny(gray, 30, 200)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in contours:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(image, image, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped_licence_plate = gray[topx:bottomx + 1, topy:bottomy + 1]

    return cropped_licence_plate
