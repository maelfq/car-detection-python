from types import NoneType
import detect_cars as cu

# Perform object detection on an image
mercedes: str = r"C:\dev\car-detection-python\resources\red-mercedes.jpeg"

def main() -> NoneType:
    carUtils: cu.CarUtils = cu.CarUtils("yolov11")
    carUtils.detectCarsOnImage(mercedes)

main()