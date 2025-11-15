from ultralytics import YOLO
import cv2

model = YOLO("runs/animal_yolo/animal_detection/weights/best.pt")

image_path = r"C:\Users\NguyenHuy\Pictures\hq720.jpg"


results = model(image_path)

for r in results:
    annotated_img = r.plot()
    cv2.imshow("Result", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
