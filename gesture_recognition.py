import cv2
import numpy as np
from hand_tracking import HandTracker

def recognize_gesture(results):
    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        return f"Detected {num_hands} hand(s)"
    return "No hands detected"

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = tracker.detect_hands(frame)
        gesture_text = recognize_gesture(results)

        cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
