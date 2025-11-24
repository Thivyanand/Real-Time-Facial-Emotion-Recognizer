import cv2
import time
from fer import FER

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

detector = FER(mtcnn=False)

cap = cv2.VideoCapture(0)

#fps
prev_time = time.time()
smooth_fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

#fps calc
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time


    smooth_fps = (smooth_fps * 0.8) + (fps * 0.2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]  #faces 1

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        
        face_roi = frame[y:y+h, x:x+w]
        results = detector.detect_emotions(face_roi)

        if results:
            emotions = results[0]["emotions"]

            
            emotion_lines = [f"{e}: {v*100:.1f}%" for e, v in emotions.items()]

            # 
            box_x, box_y = 10, 10
            box_w = 240
            box_h = 20 * len(emotion_lines) + 15

            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y),
                          (box_x + box_w, box_y + box_h),
                          (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            #
            y_offset = box_y + 20
            for line in emotion_lines:
                cv2.putText(frame, line, (box_x + 10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0), 1)
                y_offset += 18

    #fps counter
    cv2.putText(frame, f"FPS: {smooth_fps:.1f}",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 0), 1)

    cv2.imshow("Emotion Recognition (Fast + Smooth FPS)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
