import numpy as np
import cv2
import pytesseract

def extract_number_plate_text(image):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply preprocessing
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(filtered, 30, 200)  # Edge detection

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_text = "No plate detected"

    # Loop over contours to find the number plate
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:  # Possible number plate
            x, y, w, h = cv2.boundingRect(approx)
            plate_region = gray[y:y+h, x:x+w]

            # Preprocessing for OCR
            plate_region = cv2.threshold(plate_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Extract text using Tesseract
            plate_text = pytesseract.image_to_string(plate_region, config="--psm 8").strip()
            break

    return plate_text