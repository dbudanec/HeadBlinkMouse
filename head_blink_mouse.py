import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque
import math

# Disable PyAutoGUI failsafe to allow mouse control to screen edges
pyautogui.FAILSAFE = False

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# --- Constants for landmark indices ---
LEFT_EYE_LANDMARKS = (145, 159)  # Top and bottom landmarks for left eye
RIGHT_EYE_LANDMARKS = (374, 386)  # Top and bottom landmarks for right eye
NOSE_TIP_LANDMARK = 1  # Landmark index for the nose tip

# --- Tuning Parameters ---
# Normal Smoothing (for larger movements)
NORMAL_SMOOTHING_WINDOW = 1  # Number of recent positions to average for normal movement
NORMAL_ALPHA = 0.2  # Exponential smoothing factor for normal movement (0 < ALPHA <= 1)

# Precise Smoothing (for smaller, fine-tuned movements)
PRECISE_SMOOTHING_WINDOW = 15  # Smaller window for more responsiveness in precise mode
PRECISE_ALPHA = 0.05  # Higher alpha for faster reaction in precise mode

# Threshold to switch between normal and precise smoothing modes
# If the raw intended move (before smoothing) is less than this many pixels, use precise mode.
PRECISE_MODE_DISTANCE_THRESHOLD = 60  # Pixels

# Blink detection threshold (relative distance between eye landmarks)
BLINK_THRESHOLD = 0.0045  # Adjusted for typical landmark scaling

# Cooldown period after a click to prevent multiple rapid clicks
CLICK_COOLDOWN = 0.7  # seconds


def calibrate_camera(face_mesh_detector, camera):
    """
    Guides the user through a calibration process to map facial movements to screen coordinates.
    The user looks at screen corners and presses SPACE.
    Args:
        face_mesh_detector: The initialized MediaPipe FaceMesh object.
        camera: The OpenCV VideoCapture object.
    Returns:
        A tuple (min_x, max_x, min_y, max_y) representing the calibrated
        range of nose tip landmark coordinates, or None if calibration is aborted.
    """
    instructions = [
        "Look at TOP-LEFT corner & press SPACE",
        "Look at TOP-RIGHT corner & press SPACE",
        "Look at BOTTOM-LEFT corner & press SPACE",
        "Look at BOTTOM-RIGHT corner & press SPACE"
    ]

    calibration_points_landmarks = []  # Stores (landmark.x, landmark.y) from nose tip
    print("Starting camera calibration...")

    for i, msg in enumerate(instructions):
        print(f"Calibration step {i + 1}/4: {msg}")
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Error: Failed to capture frame from camera during calibration.")
                time.sleep(0.1)
                continue

            # Flip the frame horizontally for a more natural mirror view
            frame = cv2.flip(frame, 1)
            # Convert BGR image to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame to get face landmarks
            output = face_mesh_detector.process(rgb_frame)

            # Display instructions on the frame
            display_frame = frame.copy()
            cv2.putText(display_frame, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Calibration", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key
                print("Calibration aborted by user.")
                cv2.destroyWindow("Calibration")
                return None
            if key == 32:  # SPACE key
                if output.multi_face_landmarks:
                    landmarks = output.multi_face_landmarks[0].landmark
                    nose_tip = landmarks[NOSE_TIP_LANDMARK]
                    calibration_points_landmarks.append((nose_tip.x, nose_tip.y))
                    print(f"Point {i + 1} captured: ({nose_tip.x:.4f}, {nose_tip.y:.4f})")
                    break
                else:
                    print("No face detected. Please ensure your face is visible.")
                    # Show a message on screen if no face detected
                    cv2.putText(display_frame, "No face detected!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Calibration", display_frame)
                    cv2.waitKey(1)  # Allow window to update

    cv2.destroyWindow("Calibration")
    if len(calibration_points_landmarks) != 4:
        print("Error: Calibration did not complete successfully. Not enough points.")
        return None

    # Determine the min/max x and y landmark coordinates from calibration
    xs = [p[0] for p in calibration_points_landmarks]
    ys = [p[1] for p in calibration_points_landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Add a small buffer to prevent extreme sensitivity at edges
    buffer_x = (max_x - min_x) * 0.05
    buffer_y = (max_y - min_y) * 0.05
    min_x -= buffer_x
    max_x += buffer_x
    min_y -= buffer_y
    max_y += buffer_y

    print(f"Calibration complete. X-range: ({min_x:.4f}, {max_x:.4f}), Y-range: ({min_y:.4f}, {max_y:.4f})")
    return (min_x, max_x, min_y, max_y)


def main():
    # Initialize webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize FaceMesh with landmark refinement for better eye tracking
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # Enables iris and lip landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Get screen dimensions
    screen_w, screen_h = pyautogui.size()
    print(f"Screen dimensions: {screen_w}x{screen_h}")

    # --- Calibration ---
    calibration_data = calibrate_camera(face_mesh, cam)
    if not calibration_data:
        print("Calibration aborted or failed. Exiting.")
        cam.release()
        cv2.destroyAllWindows()
        return
    min_cal_x, max_cal_x, min_cal_y, max_cal_y = calibration_data

    # Deque to store recent raw target positions for Simple Moving Average (SMA)
    # Maxlen should be the larger of the two smoothing windows
    max_deque_len = max(NORMAL_SMOOTHING_WINDOW, PRECISE_SMOOTHING_WINDOW)
    recent_raw_positions = deque(maxlen=max_deque_len)

    # Variables for Exponential Moving Average (EMA) smoothed position
    exp_smoothed_x, exp_smoothed_y = None, None

    # Timestamp for click cooldown
    last_click_time = 0

    print("Starting head and eye tracking. Press ESC to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture frame.")
            time.sleep(0.1)  # Wait a bit before retrying
            continue

        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        # Get frame dimensions
        frame_h, frame_w, _ = frame.shape
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process frame
        output = face_mesh.process(rgb_frame)

        if output.multi_face_landmarks:
            # Assuming only one face
            landmarks = output.multi_face_landmarks[0].landmark

            # --- HEAD TRACKING for Mouse Movement ---
            nose_tip = landmarks[NOSE_TIP_LANDMARK]

            # Normalize nose position based on calibration data
            # Clamp nose.x and nose.y to prevent going outside calibrated range, which could lead to extreme values
            clamped_nose_x = max(min_cal_x, min(nose_tip.x, max_cal_x))
            clamped_nose_y = max(min_cal_y, min(nose_tip.y, max_cal_y))

            # Calculate relative position within the calibrated range
            # Avoid division by zero if calibration range is too small
            range_x = max_cal_x - min_cal_x
            range_y = max_cal_y - min_cal_y

            if range_x == 0 or range_y == 0:  # Should not happen with buffer in calibration
                print("Warning: Calibration range is zero for X or Y. Check calibration.")
                # Fallback to center of screen or skip movement
                raw_target_x, raw_target_y = screen_w // 2, screen_h // 2
            else:
                relative_x = (clamped_nose_x - min_cal_x) / range_x
                relative_y = (clamped_nose_y - min_cal_y) / range_y
                # Map to screen coordinates
                raw_target_x = int(relative_x * screen_w)
                raw_target_y = int(relative_y * screen_h)

            # Ensure raw target coordinates are within screen bounds
            raw_target_x = max(0, min(raw_target_x, screen_w - 1))
            raw_target_y = max(0, min(raw_target_y, screen_h - 1))

            # --- Determine Smoothing Mode (Normal or Precise) ---
            current_mouse_pos = pyautogui.position()
            active_smoothing_window = NORMAL_SMOOTHING_WINDOW
            active_alpha = NORMAL_ALPHA

            if current_mouse_pos:
                current_screen_x, current_screen_y = current_mouse_pos
                # Calculate distance between raw target and current mouse position
                intended_move_dist = math.hypot(raw_target_x - current_screen_x, raw_target_y - current_screen_y)

                if intended_move_dist <= PRECISE_MODE_DISTANCE_THRESHOLD:
                    active_smoothing_window = PRECISE_SMOOTHING_WINDOW
                    active_alpha = PRECISE_ALPHA
            # If pyautogui.position() fails, defaults to normal smoothing (already set)

            # --- Apply Smoothing ---
            # 1) Simple Moving Average (SMA) on raw target positions
            recent_raw_positions.append((raw_target_x, raw_target_y))

            # Use the active_smoothing_window for SMA
            num_samples_for_avg = min(len(recent_raw_positions), active_smoothing_window)

            if num_samples_for_avg > 0:
                # Get the most recent 'num_samples_for_avg' samples
                current_window_samples = list(recent_raw_positions)[-num_samples_for_avg:]
                sma_x = sum(p[0] for p in current_window_samples) / num_samples_for_avg
                sma_y = sum(p[1] for p in current_window_samples) / num_samples_for_avg
            else:  # Should only happen at the very beginning if deque is empty
                sma_x, sma_y = raw_target_x, raw_target_y

            # 2) Exponential Moving Average (EMA) on the SMA result
            if exp_smoothed_x is None or exp_smoothed_y is None:
                # Initialize EMA with the first SMA value
                exp_smoothed_x, exp_smoothed_y = sma_x, sma_y
            else:
                # Apply EMA using active_alpha
                exp_smoothed_x = active_alpha * sma_x + (1 - active_alpha) * exp_smoothed_x
                exp_smoothed_y = active_alpha * sma_y + (1 - active_alpha) * exp_smoothed_y

            # Move mouse to the final smoothed position
            pyautogui.moveTo(int(exp_smoothed_x), int(exp_smoothed_y), duration=0)

            # --- EYE BLINK DETECTION for Clicks ---
            # Get vertical distance between top and bottom landmarks for each eye
            # (Using landmark.y directly; no need for frame_h scaling if threshold is relative)

            # Left Eye
            left_eye_top_y = landmarks[LEFT_EYE_LANDMARKS[0]].y
            left_eye_bottom_y = landmarks[LEFT_EYE_LANDMARKS[1]].y
            left_eye_dist = abs(left_eye_top_y - left_eye_bottom_y)

            # Right Eye
            right_eye_top_y = landmarks[RIGHT_EYE_LANDMARKS[0]].y
            right_eye_bottom_y = landmarks[RIGHT_EYE_LANDMARKS[1]].y
            right_eye_dist = abs(right_eye_top_y - right_eye_bottom_y)

            current_time = time.time()
            if current_time - last_click_time > CLICK_COOLDOWN:
                # Determine blink type
                left_blink = left_eye_dist < BLINK_THRESHOLD
                right_blink = right_eye_dist < BLINK_THRESHOLD

                if left_blink and not right_blink:  # Left eye blink only
                    pyautogui.click(button='left')
                    print("Left Click (Left Eye Blink)")
                    last_click_time = current_time
                    # Optional: Visual feedback for blink
                    cv2.circle(frame, (50, frame_h - 50), 20, (0, 255, 0), -1)
                elif right_blink and not left_blink:  # Right eye blink only
                    pyautogui.click(button='right')
                    print("Right Click (Right Eye Blink)")
                    last_click_time = current_time
                    cv2.circle(frame, (frame_w - 50, frame_h - 50), 20, (0, 0, 255), -1)
                # elif left_blink and right_blink: # Both eyes blink
                #     print("Both Eyes Blinked - No action")
                #     pass # Or assign another action

            # --- Visualization (Optional) ---
            # Draw nose tip on frame
            cv2.circle(frame, (int(nose_tip.x * frame_w), int(nose_tip.y * frame_h)), 3, (0, 255, 0), -1)
            # Draw eye landmarks (example for left eye)
            cv2.circle(frame, (int(landmarks[LEFT_EYE_LANDMARKS[0]].x * frame_w),
                               int(landmarks[LEFT_EYE_LANDMARKS[0]].y * frame_h)), 2, (255, 0, 0), -1)
            cv2.circle(frame, (int(landmarks[LEFT_EYE_LANDMARKS[1]].x * frame_w),
                               int(landmarks[LEFT_EYE_LANDMARKS[1]].y * frame_h)), 2, (255, 0, 0), -1)

        # Display the frame
        cv2.imshow('Head and Eye Control', frame)

        # Check for ESC key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("ESC pressed. Exiting.")
            break

    # Release resources
    cam.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Application closed.")


if __name__ == "__main__":
    main()
