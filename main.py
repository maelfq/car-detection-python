from types import NoneType
from detect_cars.detect_cars import CarUtils

# Perform object detection on an image
# mercedes: str = r"C:\dev\car-detection-python\resources\red-mercedes.jpeg"
car: str = r'C:\dev\car-detection-python\00d0c3e342d41462.jpg'
def main() -> NoneType:
    carUtils: CarUtils = CarUtils("yolov11")
    carUtils.detect_cars_in_image(car)

if __name__ == '__main__':
    main()