import cv2
import imutils
import numpy as np
from cv2 import Mat
from numpy import ndarray
import easyocr


def detect_licence_plate(image: Mat | ndarray) -> str:
    licence_plate_image = get_licence_plate_cropped(image)
    imageTmp = cv2.imwrite(r'C:\dev\car-detection-python\tmp.jpeg', licence_plate_image)
    texts = read_text_from_image(imageTmp)
    if len(texts) >= 1:
        return texts[0]
    return ""


def read_text_from_image(image: Mat | ndarray) -> list[str]:
    tmpImagePath: str = r'C:\dev\car-detection-python\tmp.jpeg'
    cv2.imwrite(tmpImagePath, image)
    reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
    result = reader.readtext(tmpImagePath)

    predicted_license_plates = []

    for detection in result:
        print(detection)
        predicted_license_plates.append(detection[1])

    if len(predicted_license_plates) == 1:
        save_to_database(predicted_license_plates[0])

    return predicted_license_plates


def save_to_database(licence_plate: str) -> bool:
    # TODO
    return False


def get_licence_plate_cropped(image: Mat | ndarray) -> Mat | ndarray:
    # code found here: https://cyberworrier2000.medium.com/license-plate-recognition-using-opencv-python-e03dd591f083

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
    Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

    return Cropped
