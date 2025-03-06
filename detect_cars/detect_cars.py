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
    enable_draw_boxes: bool = False

    def __init__(self, name):
        self.model = YOLO("yolo11n.pt")
        self.name = name

    def detect_cars_in_image(self, image_path: str) -> NoneType:

        image: Mat | ndarray = cv2.imread(image_path)

        if self.is_picture_too_big(image_path):
            image = self.resize_image(image_path)
            image_path = self.get_resized_image_path(image_path)

        coordinates_of_cars = self.get_car_boxes(image_path)
        if self.enable_draw_boxes:
            self.draw_boxes_on_image(coordinates_of_cars, image_path)

        for car_index in range(len(coordinates_of_cars)):
            # crop image of car
            car_image = image[coordinates_of_cars[car_index][1]:coordinates_of_cars[car_index][3], coordinates_of_cars[car_index][0]:coordinates_of_cars[car_index][2]]
            licence_plate_text: str = detect_licence_plate(car_image)

        return


    def get_car_boxes(self, imagePath: str) -> list[list[int]]:
        results = self.model(imagePath)
        boxes = []
        for result in results:
            if self.model.names[int(result.boxes.cls[0])] == 'car':
                tuple_coordinates: tuple = result.boxes.xyxy.numpy()[0].tolist()
                rounded_coordinates = []
                for coord_index in range(len(tuple_coordinates)):
                    rounded_coordinates.append(math.floor(tuple_coordinates[coord_index]))
                boxes.append(rounded_coordinates)
        print(boxes)
        return boxes

    def draw_boxes_on_image(self, coordinates: list[list[int]], image_path: str) -> None:
        image = cv2.imread(image_path)

        for box in coordinates:
            print(box)
            startPoint = (box[0], box[1])
            endPoint = (box[2], box[3])
            thickness = 2
            color = (100, 100, 100)

            image = cv2.rectangle(image, startPoint, endPoint, color, thickness)
        cv2.imwrite(self.get_boxes_image_path(image_path), image)

        return

    def is_picture_too_big(self, image_path: str) -> bool:

        image = cv2.imread(image_path)

        max_height: int = 1080
        max_width: int = 1920

        width: int = image.shape[0]
        height: int = image.shape[1]
        if width > max_height or height > max_width:
            return True

        return False

    def get_resized_image_shape(self, image_path: str) -> tuple[int, int]:

        image = cv2.imread(image_path)

        max_height: int = 1080
        max_width: int = 1920

        height: int = image.shape[0]
        width: int = image.shape[1]

        new_width: int
        new_height: int

        correct_width_factor: float = max_width / width
        correct_height_factor: float = max_height / height

        factor: float = correct_width_factor if (correct_width_factor < correct_height_factor) else correct_height_factor

        new_height = math.floor(factor * height)
        new_width = math.floor(factor * width)

        return new_height, new_width

    def get_resized_image_path(self, image_path: str) -> str:
        file_name: str = str(os.path.basename(image_path))
        directory: str = str(os.path.dirname(image_path))

        return directory + os.sep + 'resized_' + file_name

    def get_boxes_image_path(self, image_path) -> str:
        file_name: str = str(os.path.basename(image_path))
        directory: str = str(os.path.dirname(image_path))

        return directory + os.sep + 'boxes_' + file_name

    def resize_image(self, image_path: str) -> Mat | ndarray:
        image = cv2.imread(image_path)

        resized_image_path: str = self.get_resized_image_path(image_path)

        new_height, new_width = self.get_resized_image_shape(image_path)
        new_image = cv2.resize(image, (new_width, new_height))
        cv2.imwrite(resized_image_path, new_image)

        return new_image
