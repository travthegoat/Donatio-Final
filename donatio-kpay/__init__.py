from kpay_processor import KPaySlipProcessor
import os 

os.chdir("D:\\donatio-AI\\src\\kpay_detect\\data")
processor = KPaySlipProcessor("kpay.jpg")
data = processor.process()