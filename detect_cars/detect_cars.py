import math
import os.path
from types import NoneType

import cv2
from cv2 import Mat
from numpy import ndarray, dtype
from ultralytics import YOLO

from detect_cars.detect_licence_plate import detect_licence_plate


class CarUtils:
    model: YOLO | NoneType = None
    name: str = "None"

    def __init__(self, name):
        self.model = YOLO("yolo11n.pt")
        self.name = name

    def detect_cars_in_image(self, imagePath: str) -> NoneType:

        image: Mat | ndarray = cv2.imread(imagePath)

        if self.is_picture_too_big(imagePath):
            image = self.resize_image(imagePath)
            imagePath = self.get_resized_image_path(imagePath)

        coordinatesOfCars = self.get_car_boxes(imagePath)
        self.draw_boxes_on_image(coordinatesOfCars, imagePath)

        print('yo')
        for carIndex in range(len(coordinatesOfCars)):
            # crop image of car
            carImage = image[coordinatesOfCars[carIndex][1]:coordinatesOfCars[carIndex][3], coordinatesOfCars[carIndex][0]:coordinatesOfCars[carIndex][2]]
            # cv2.imwrite('C:\\dev\\car-detection-python\\test2.jpeg', carImage)

            detect_licence_plate(carImage)

        return


    def get_car_boxes(self, imagePath: str) -> list[list[int]]:
        results = self.model(imagePath)
        boxes = []
        for result in results:
            if self.model.names[int(result.boxes.cls[0])] == 'car':
                tupleCoordinates: tuple = result.boxes.xyxy.numpy()[0].tolist()
                roundedCoordinates = []
                for coordIndex in range(len(tupleCoordinates)):
                    roundedCoordinates.append(math.floor(tupleCoordinates[coordIndex]))
                boxes.append(roundedCoordinates)
        print(boxes)
        return boxes

    def draw_boxes_on_image(self, coordinates: list[list[int]], imagePath: str) -> None:
        image = cv2.imread(imagePath)

        for box in coordinates:
            print(box)
            startPoint = (box[0], box[1])
            endPoint = (box[2], box[3])
            thickness = 2
            color = (100, 100, 100)
            print(startPoint)
            print(endPoint)

            image = cv2.rectangle(image, startPoint, endPoint, color, thickness)
        cv2.imwrite(self.get_boxes_image_path(imagePath), image)

        return

    def is_picture_too_big(self, imagePath: str) -> bool:

        image = cv2.imread(imagePath)

        maxHeight: int = 1080
        maxWidth: int = 1920

        width: int = image.shape[0]
        height: int = image.shape[1]
        if width > maxHeight or height > maxWidth:
            return True

        return False

    def get_resized_image_shape(self, imagePath: str) -> tuple[int, int]:

        image = cv2.imread(imagePath)

        maxHeight: int = 1080
        maxWidth: int = 1920

        height: int = image.shape[0]
        width: int = image.shape[1]

        newWidth: int
        newHeight: int

        correctWidthFactor: float = maxWidth / width
        correctHeightFactor: float = maxHeight / height

        factor: float = correctWidthFactor if (correctWidthFactor < correctHeightFactor) else correctHeightFactor

        newHeight = math.floor(factor * height)
        newWidth = math.floor(factor * width)

        return newHeight, newWidth

    def get_resized_image_path(self, imagePath: str) -> str:
        fileName: str = str(os.path.basename(imagePath))
        directory: str = str(os.path.dirname(imagePath))

        return directory + os.sep + 'resized_' + fileName

    def get_boxes_image_path(self, imagePath) -> str:
        fileName: str = str(os.path.basename(imagePath))
        directory: str = str(os.path.dirname(imagePath))

        return directory + os.sep + 'boxes_' + fileName

    def resize_image(self, imagePath: str) -> Mat | ndarray:
        image = cv2.imread(imagePath)

        resizedImagePath: str = self.get_resized_image_path(imagePath)

        newHeight, newWidth = self.get_resized_image_shape(imagePath)
        newImage = cv2.resize(image, (newWidth, newHeight))
        cv2.imwrite(resizedImagePath, newImage)

        return newImage
