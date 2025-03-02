import math
import os.path
from types import NoneType
import cv2
from ultralytics import YOLO


class CarUtils:
    model: YOLO | NoneType = None
    name: str = "None"

    def __init__(self, name):
        print('init')
        self.model = YOLO("yolo11n.pt")
        self.name = name

    def detectCarsOnImage(self, imagePath: str) -> NoneType:

        if self.isPictureTooBig(imagePath):
            resizedImagePath: str = self.resizeImage(imagePath)
            if os.path.exists(resizedImagePath):
                imagePath = resizedImagePath

        #TODO: resize image and save it, returns resized image path
        coordinatesOfCar = self.getCarBoxes(imagePath)
        print(imagePath)
        self.drawBoxesOnImage(coordinatesOfCar, imagePath)
        #TODO: detect licence plates
        # -> https://pyimagesearch.com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python/
        #TODO: save licence plates in DB
        return

    def getCarBoxes(self, imagePath: str) -> list[list[int]]:
        results = self.model(imagePath)
        print(results[0].boxes.xyxy)
        boxes = []
        for result in results:
            if self.model.names[int(result.boxes.cls[0])] == 'car':
                print('Car detected')
                tuple: tuple = result.boxes.xyxy.numpy()[0].tolist()
                boxes.append(tuple)
        print(boxes)
        return boxes

    def drawBoxesOnImage(self, coordinates: list[list[int]], imagePath: str) -> None:
        print('drawBoxes')
        image = cv2.imread(imagePath)

        for box in coordinates:
            print(box)
            startPoint = (math.floor(box[0]), math.floor(box[1]))
            endPoint = (math.floor(box[2]), math.floor(box[3]))
            thickness = 2
            color = (100, 100, 100)
            print(startPoint)
            print(endPoint)

            image = cv2.rectangle(image, startPoint, endPoint, color, thickness)
        cv2.imwrite('test1.jpeg', image)

        return

    def isPictureTooBig(self, imagePath: str) -> bool:

        image = cv2.imread(imagePath)

        maxHeight: int = 1080
        maxWidth: int = 1920

        width: int = image.shape[0]
        height: int = image.shape[1]
        if width > maxHeight or height > maxWidth:
            return True

        return False

    def getResizedImageShape(self, imagePath: str) -> tuple[int, int]:

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

    def resizeImage(self, imagePath: str) -> str:

        fileName:str = str(os.path.basename(imagePath))
        directory: str = str(os.path.dirname(imagePath))

        image = cv2.imread(imagePath)

        resizedImagePath: str = directory + os.sep + 'resized_' + fileName

        newHeight, newWidth = self.getResizedImageShape(imagePath)
        newImage = cv2.resize(image, (newWidth, newHeight))
        cv2.imwrite(resizedImagePath, newImage)

        return resizedImagePath
