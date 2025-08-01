import cv2
import easyocr
import numpy as np
import re
import os 

class KPaySlipProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.reader = easyocr.Reader(['en'])
        self.img = None
        self.cropped_img = None
        self.results = []

    def load_image(self):
        image = cv2.imread(self.image_path)
        self.img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def reorder_points(self, points):
        points = points.reshape((4, 2))
        ordered = np.zeros((4, 1, 2), dtype=np.int32)
        add = points.sum(1)
        diff = np.diff(points, axis=1)

        ordered[0] = points[np.argmin(add)]
        ordered[3] = points[np.argmax(add)]
        ordered[1] = points[np.argmin(diff)]
        ordered[2] = points[np.argmax(diff)]
        return ordered

    def find_valid_contour(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 3)
        thresh = cv2.threshold(blurred, 104, 150, cv2.THRESH_BINARY)[1]
        filtered = cv2.bilateralFilter(thresh, 1, 188, 260)
        edges = cv2.Canny(filtered, 1, 10, apertureSize=3)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 8000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.6 < aspect_ratio < 1.4 and w > 100 and h > 100:
                        return self.reorder_points(approx[:4])
        return None

    def crop_by_contour(self, rect):
        coords = rect.reshape(-1, 2)
        x1, x2 = max(0, np.min(coords[:, 0])), min(self.img.shape[1], np.max(coords[:, 0]))
        y1, y2 = max(0, np.min(coords[:, 1])), min(self.img.shape[0], np.max(coords[:, 1]))
        self.cropped_img = self.img[y1:y2, x1:x2]

    def perform_ocr(self):
        if self.cropped_img is not None:
            bgr = cv2.cvtColor(self.cropped_img, cv2.COLOR_BGR2RGB)
        else:
            bgr = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.results = self.reader.readtext(bgr)

    def extract_values(self):
        try:
            name_and_phone = self.results[11][1]
            amount = self.results[13][1]

            clean_amount = int(re.sub(r'\.00$', '', re.sub(r'[^\d.]', '', amount)))
            name, phone = re.match(r'^(.*?)\s*\((.*?)\)$', name_and_phone).groups()

            return {
                "clean_amount": clean_amount,
                "name": name,
                "phone": phone
            }
        except Exception as e:
            print("Extraction failed:", e)
            print("OCR Results:", [r[1] for r in self.results])
            return None

    def process(self):
        self.load_image()
        rect = self.find_valid_contour()

        if rect is not None:
            self.crop_by_contour(rect)
        else:
            pass

        self.perform_ocr()
        values = self.extract_values()

        if values:
            print("Clean Amount:", values["clean_amount"])
            print("Name:", values["name"])
            print("Phone:", values["phone"])
        else:
            print("Failed to extract structured data.")

        return values

if __name__ == "__main__":
    os.chdir("D:\\donatio-AI\\src\\kpay_detect\\data")
    processor = KPaySlipProcessor("kpay.jpg")
    data = processor.process()
