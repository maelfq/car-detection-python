import os.path
import unittest
from detect_cars import CarUtils


class CarUtilsTest(unittest.TestCase):
    mercedes: str = r"C:\dev\car-detection-python\resources\red-mercedes.jpeg"

    def test_getCarBoxesTest(self):
        # arrange
        carUtils = CarUtils("Test")

        # act
        boxes: list[list[int]] = carUtils.getCarBoxes(self.mercedes)

        # assert
        self.assertTrue(1 == len(boxes))

    def test_isPictureTooBig(self):
        # arrange
        carUtils = CarUtils("Test")

        # act
        isPictureTooBig = carUtils.isPictureTooBig(self.mercedes)

        # assert
        self.assertTrue(isPictureTooBig)

    def test_getResizedImageShape(self):
        # arrange
        carUtils = CarUtils("Test")

        # act
        newSize = carUtils.getResizedImageShape(self.mercedes)

        # assert
        self.assertEqual((1080, 1622), newSize)


    def test_resizeImage(self):
        # arrange
        carUtils = CarUtils("Test")

        # act
        resizedPath: str = carUtils.resizeImage(self.mercedes)

        # assert
        self.assertTrue(os.path.exists(resizedPath))

        # TODO: assert new image size


if __name__ == '__main__':
    unittest.main()
