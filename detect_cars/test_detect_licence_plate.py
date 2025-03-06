import unittest
import os
import cv2

import detect_licence_plate

class DetectLicencePlateTest(unittest.TestCase):
    def test_read_text_from_image(self):
        licence_plate_picture_path: str = os.getcwd() + os.sep + 'resources' + os.sep + '0c89e35ae1906abf.jpg'

        # arrange
        image = cv2.imread(licence_plate_picture_path)

        # act
        licence_plates = detect_licence_plate.read_text_from_image(image)
        found: bool = 'LOLZOMG' in licence_plates

        #assert
        self.assertTrue(found)


if __name__ == '__main__':
    unittest.main()
