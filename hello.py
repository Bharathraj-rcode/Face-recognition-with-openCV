import cv2
from simple_facrec import SimpleFacerec

sfr = SimpleFacerec()
sfr.load_encoding_images("C:/Users/HP/Desktop/my_python/Face_Recognition/img")

cap = cv2.VideoCapture(0)


while True:
    ret , frame = cap.read()
    face_location , face_name = sfr.detect_known_faces(frame)
    for loc, name in zip(face_location, face_name):
        y1, x2, y2, x1 = loc  # Note: usually face_recognition returns (top, right, bottom, left)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)


        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,200),2)
    cv2.imshow("FRAME:" , frame)
    key = cv2.waitKey(5)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
