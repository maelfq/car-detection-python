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

        #TODO: resize image and save it, returns resized image path
        coordinatesOfCar = self.getCarBoxes(imagePath)
        self.drawBoxesOnImage(coordinatesOfCar,imagePath)
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
        image = cv2.imread(imagePath)
        for box in coordinates:
            print(box)
            startPoint = (int(box[0]), int(box[2]))
            endPoint = (int(box[1]),int(box[3]))
            thickness = 2
            color = (255,0,0)
            image = cv2.rectangle(image, startPoint, endPoint, color, thickness)
        cv2.imshow("test", image)
        return
