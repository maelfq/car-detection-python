import unittest

import cv2

import detect_licence_plate

class DetectLicencePlateTest(unittest.TestCase):
    def test_read_text_from_image(self):
        licencePlatePicturePath: str = r'C:\dev\car-detection-python\resources\0c89e35ae1906abf.jpg'

        # arrange
        image = cv2.imread(licencePlatePicturePath)

        # act
        licencePlates = detect_licence_plate.read_text_from_image(image)
        found: bool = 'LOLZOMG' in licencePlates

        #assert
        self.assertTrue(found)


if __name__ == '__main__':
    unittest.main()
