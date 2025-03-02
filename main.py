from types import NoneType
from detect_cars.detect_cars import CarUtils

# Perform object detection on an image
mercedes: str = r"C:\dev\car-detection-python\resources\red-mercedes.jpeg"

def main() -> NoneType:
    carUtils: CarUtils = CarUtils("yolov11")
    carUtils.detectCarsOnImage(mercedes)

if __name__ == '__main__':
    main()