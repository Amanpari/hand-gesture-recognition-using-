import cv2
import mediapipe as mp
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.9, track_confidence=0.9):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=track_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def detect_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)

    def get_palm_center(self, hand_landmarks, frame):
        h, w, _ = frame.shape
        ids = [
            self.mp_hands.HandLandmark.WRIST,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP,
        ]
        points = [(int(hand_landmarks.landmark[i].x * w), int(hand_landmarks.landmark[i].y * h)) for i in ids]
        center = np.mean(points, axis=0)
        return center

    def classify_gesture(self, results, frame):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers = []
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

                if thumb_tip.y < wrist.y - 0.05:
                    fingers.append(1)
                elif thumb_tip.y > wrist.y + 0.04:
                    fingers.append(-1)
                else:
                    fingers.append(0)

                finger_tips = [
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    self.mp_hands.HandLandmark.RING_FINGER_TIP,
                    self.mp_hands.HandLandmark.PINKY_TIP,
                ]
                finger_pips = [
                    self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                    self.mp_hands.HandLandmark.RING_FINGER_PIP,
                    self.mp_hands.HandLandmark.PINKY_PIP,
                ]

                for tip, pip in zip(finger_tips, finger_pips):
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y - 0.02:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                gesture = "Unknown"

                dist_ok = np.linalg.norm(
                    np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y])
                )

                if dist_ok < 0.05 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                    gesture = "OK"
                elif fingers == [1, 1, 1, 1, 1]:
                    gesture = "Open Palm"
                elif fingers[1] == 1 and sum(fingers[2:]) == 0:
                    gesture = "Pointing"
                elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers[3:]) == 0:
                    gesture = "Peace Sign"
                elif fingers == [1, 0, 0, 0, 0]:
                    gesture = "Thumbs Up"
                elif fingers == [-1, 0, 0, 0, 0]:
                    gesture = "Thumbs Down"
                elif fingers == [1, 1, 0, 0, 1]:
                    gesture = "Rock"
                elif fingers == [1, 0, 0, 0, 1]:
                    gesture = "Call "
                elif sum(fingers) == 0:
                    gesture = "Fist"

                return gesture
        return "No Hand Detected"

    def draw_tracking_lines(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )

    def draw_convex_hull(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = []
                for landmark in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    points.append((x, y))
                points = np.array(points, dtype=np.int32)
                if len(points) > 0:
                    hull = cv2.convexHull(points)
                    cv2.polylines(frame, [hull], isClosed=True, color=(0, 255, 255), thickness=3)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))
        results = tracker.detect_hands(frame)
        gesture = tracker.classify_gesture(results, frame)

        tracker.draw_tracking_lines(frame, results)
        tracker.draw_convex_hull(frame, results)

        cv2.putText(frame, f"Gesture: {gesture}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
