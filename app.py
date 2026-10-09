"""
Phase 5: Gesture Control System Integration
Integrates HandTracker, GestureClassifier, SlideTracker, and OSController.
Full end-to-end functionality:
- Pinch-and-Slide: variable-speed smooth scrolling
- Peace Sign: debounced Play/Pause media toggle
- Spacebar / 't' key: toggle gesture control ON/OFF
"""

import time
import cv2
from tracker import HandTracker
from gestures import GestureClassifier
from motion import SlideTracker
from controller import OSController


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    tracker = HandTracker(model_path="models/hand_landmarker.task", max_hands=1)
    classifier = GestureClassifier(pinch_threshold_ratio=0.28)
    slide_tracker = SlideTracker(deadzone_ratio=0.04, speed_factor=2.5, smoothing_alpha=0.6)
    os_controller = OSController(media_cooldown_seconds=1.2, scroll_multiplier=1.2)

    print("=" * 60)
    print(" GESTURE CONTROL RUNNING - PHASE 5")
    print("=" * 60)
    print(" Controls:")
    print("  • Pinch & Slide: Touchscreen-style variable scrolling")
    print("  • Peace Sign (V): Play / Pause (1.2s debounced)")
    print("  • Press 't' in window: Toggle tracking ON / OFF")
    print("  • Press 'q' or 'Esc': Exit")
    print("  • Emergency Fail-Safe: Slam mouse pointer into any screen corner")
    print("=" * 60)

    start_time = time.time()
    prev_time = time.time()
    fps = 0.0
    action_banner = ""
    banner_timer = 0.0

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break

        frame = cv2.flip(frame, 1)
        timestamp_ms = int((time.time() - start_time) * 1000)

        tracker.process_frame(frame, timestamp_ms)
        landmarks_px, _, hand_scale = tracker.get_landmarks(frame.shape)

        scroll_delta = 0.0
        velocity = 0.0
        current_pose = "NO_HAND"
        is_pinched = False
        pinch_center = None

        if landmarks_px:
            tracker.draw_skeleton(frame, landmarks_px)

            is_pinched, norm_dist, pinch_center = classifier.check_pinch(
                landmarks_px, hand_scale
            )
            finger_states = classifier.get_finger_states(landmarks_px)
            current_pose = classifier.classify_pose(finger_states, is_pinched)

            # 1. Slide & Scroll Action
            scroll_delta, velocity = slide_tracker.update(
                is_pinched, pinch_center, hand_scale
            )
            if abs(scroll_delta) > 0.5 and os_controller.is_enabled:
                os_controller.dispatch_scroll(scroll_delta)

            # 2. Discrete Pose Action (Play / Pause)
            if current_pose == "PLAY_PAUSE" and os_controller.is_enabled:
                triggered = os_controller.trigger_play_pause()
                if triggered:
                    action_banner = "PLAY / PAUSE TRIGGERED"
                    banner_timer = time.time() + 1.2

            # Visual overlay for pinch
            if pinch_center and is_pinched:
                arrow_length = int(velocity * 4)
                arrow_color = (0, 255, 0) if velocity >= 0 else (0, 165, 255)
                end_point = (pinch_center[0], pinch_center[1] - arrow_length)

                if abs(arrow_length) > 3:
                    cv2.arrowedLine(
                        frame, pinch_center, end_point, arrow_color, 3, tipLength=0.3
                    )

                cv2.circle(frame, pinch_center, 12, (0, 255, 0), -1, cv2.LINE_AA)
            elif pinch_center:
                cv2.circle(frame, pinch_center, 8, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            slide_tracker.reset()

        # FPS calculation
        curr_time = time.time()
        time_diff = curr_time - prev_time
        prev_time = curr_time
        if time_diff > 0:
            current_fps = 1.0 / time_diff
            fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

        # Visual HUD
        # Active Status Header
        status_text = "STATUS: ACTIVE" if os_controller.is_enabled else "STATUS: PAUSED ('t' to resume)"
        status_color = (0, 255, 0) if os_controller.is_enabled else (0, 0, 255)
        cv2.putText(
            frame,
            status_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Pose: {current_pose}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )

        # Scroll output
        dir_text = "Scroll: IDLE"
        if scroll_delta > 1.0:
            dir_text = f"Scroll: UP (+{int(scroll_delta)})"
        elif scroll_delta < -1.0:
            dir_text = f"Scroll: DOWN ({int(scroll_delta)})"

        cv2.putText(
            frame,
            dir_text,
            (15, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255) if abs(scroll_delta) > 1.0 else (180, 180, 180),
            2,
            cv2.LINE_AA,
        )

        # Cooldown progress ring / bar for Media Control
        cd_ratio = os_controller.get_cooldown_progress()
        if cd_ratio > 0.0:
            bar_w = int(120 * (1.0 - cd_ratio))
            cv2.putText(
                frame,
                "Media Cooldown:",
                (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(frame, (155, 115), (155 + 120, 127), (60, 60, 60), -1)
            cv2.rectangle(frame, (155, 115), (155 + bar_w, 127), (0, 165, 255), -1)

        # Action Toast Banner
        if time.time() < banner_timer:
            cv2.rectangle(
                frame,
                (120, 210),
                (520, 260),
                (0, 200, 0),
                -1,
            )
            cv2.putText(
                frame,
                action_banner,
                (140, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (frame.shape[1] - 110, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "Phase 5: Live Control ['t': Pause/Resume | 'q': Quit]",
            (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Gesture Control - Phase 5", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('t') or key == ord(' '):
            os_controller.toggle_active()
            print(f"[STATUS] Controller active state: {os_controller.is_enabled}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
