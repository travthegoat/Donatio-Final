from PIL import Image
from pyzbar.pyzbar import decode


def extract_qr_url(file):
    try:
        image = Image.open(file).convert("RGB")
        qr_data = decode(image)

        if qr_data:
            return qr_data[0].data.decode("utf-8")
        else:
            return None
    except Exception as e:
        return None
