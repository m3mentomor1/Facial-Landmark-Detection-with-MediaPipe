import cv2
import mediapipe as mp

# Initialize MediaPipe face detection and landmark model
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def detect_landmarks(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe face mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        # Process the image
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face landmarks on the image
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                )

    return image

# Main function
def main():
    # Open the video capture device
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Detect landmarks
        frame_with_landmarks = detect_landmarks(frame)

        # Display the resulting frame
        cv2.imshow('Face Landmark Detection', frame_with_landmarks)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
