"""
Live Gesture Control Application
Wires together tracker, gestures, motion, controller, and centralized config.
"""

import time
import cv2
import config
from tracker import HandTracker
from gestures import GestureClassifier
from motion import SlideTracker
from controller import OSController


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {config.CAMERA_INDEX}.")
        return

    tracker = HandTracker(
        model_path=config.MODEL_PATH,
        max_hands=1,
        detection_confidence=config.DETECTION_CONFIDENCE,
        tracking_confidence=config.TRACKING_CONFIDENCE,
    )
    classifier = GestureClassifier(
        pinch_in_ratio=config.PINCH_IN_RATIO,
        pinch_out_ratio=config.PINCH_OUT_RATIO,
        finger_straightness_min=config.FINGER_STRAIGHTNESS_MIN,
    )
    slide_tracker = SlideTracker(
        deadzone_ratio=config.DEADZONE_RATIO,
        speed_factor=config.BASE_SPEED_FACTOR,
        smoothing_alpha=config.SMOOTHING_ALPHA,
        flick_acceleration=config.FLICK_ACCELERATION,
        flick_min_velocity=config.FLICK_MIN_VELOCITY,
        friction_decay=config.FRICTION_DECAY,
        ghost_frame_tolerance=config.GHOST_FRAME_TOLERANCE,
    )
    os_controller = OSController(
        media_cooldown_seconds=config.MEDIA_COOLDOWN_SECONDS,
        scroll_multiplier=config.SCROLL_MULTIPLIER,
        point_scroll_speed=config.POINT_SCROLL_SPEED,
        point_hold_duration_seconds=config.POINT_HOLD_DELAY_SECONDS,
    )

    print("=" * 60)
    print(" GESTURE CONTROL RUNNING")
    print("=" * 60)
    print(" Controls:")
    print("  • Pinch & Slide: Touchscreen-style variable scrolling")
    print("  • Pinch & Flick: High-speed flick with inertia coasting")
    print("  • Point UP (Index up, others folded): Hands-free continuous scroll UP")
    print("  • Point DOWN (Index + Pinky up, others folded): Hands-free continuous scroll DOWN")
    print("  • Peace Sign (V): Play / Pause (debounced)")
    print("  • Fist: Complete idle / neutral state (cancels inputs)")
    print(f"  • Press '{config.TOGGLE_KEY}' in window: Toggle tracking ON / OFF")
    print(f"  • Press '{config.QUIT_KEY}' or 'Esc': Exit")
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

        if config.MIRROR_FEED:
            frame = cv2.flip(frame, 1)

        timestamp_ms = int((time.time() - start_time) * 1000)

        tracker.process_frame(frame, timestamp_ms)
        landmarks_px, _, hand_scale = tracker.get_landmarks(frame.shape)

        scroll_delta = 0.0
        velocity = 0.0
        is_coasting = False
        point_scrolling = False
        point_dir = None
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

            # If the user makes a FIST, override pinch state and reset motion
            if current_pose == "FIST":
                is_pinched = False
                slide_tracker.reset()

            # 1. Slide & Scroll Action (Pinch / Flick)
            scroll_delta, velocity, is_coasting = slide_tracker.update(
                is_pinched, pinch_center, hand_scale
            )
            if abs(scroll_delta) > 0.5 and os_controller.is_enabled:
                os_controller.dispatch_scroll(scroll_delta)

            # 2. Hands-Free Point Scroll (Point UP / Point DOWN)
            if not is_pinched and not is_coasting:
                point_scrolling, point_dir = os_controller.handle_point_scroll(current_pose)

            # 3. Discrete Pose Action (Play / Pause)
            if current_pose == "PLAY_PAUSE" and os_controller.is_enabled:
                triggered = os_controller.trigger_play_pause()
                if triggered:
                    action_banner = "PLAY / PAUSE TRIGGERED"
                    banner_timer = time.time() + config.MEDIA_COOLDOWN_SECONDS

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
        status_text = "STATUS: ACTIVE" if os_controller.is_enabled else f"STATUS: PAUSED ('{config.TOGGLE_KEY}' to resume)"
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

        pose_color = (255, 200, 0)
        if current_pose == "FIST":
            pose_color = (160, 160, 160)  # Neutral grey for Fist
        elif current_pose == "PLAY_PAUSE":
            pose_color = (255, 128, 0)

        cv2.putText(
            frame,
            f"Pose: {current_pose}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            pose_color,
            2,
            cv2.LINE_AA,
        )

        # Scroll output
        dir_text = "Scroll: IDLE"
        scroll_color = (180, 180, 180)

        if point_scrolling:
            dir_text = f"POINT SCROLL: {point_dir}"
            scroll_color = (0, 255, 128)  # Bright spring green for hands-free point scroll
        elif is_coasting:
            dir_text = f"FLICK GLIDE: {'UP' if scroll_delta > 0 else 'DOWN'} ({int(scroll_delta)})"
            scroll_color = (0, 165, 255)  # Orange during flick glide
        elif scroll_delta > 1.0:
            dir_text = f"Scroll: UP (+{int(scroll_delta)})"
            scroll_color = (0, 255, 255)
        elif scroll_delta < -1.0:
            dir_text = f"Scroll: DOWN ({int(scroll_delta)})"
            scroll_color = (0, 255, 255)

        cv2.putText(
            frame,
            dir_text,
            (15, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            scroll_color,
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
            f"Live Control ['{config.TOGGLE_KEY}': Pause/Resume | '{config.QUIT_KEY}': Quit]",
            (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Gesture Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(config.QUIT_KEY) or key == 27:
            break
        elif key == ord(config.TOGGLE_KEY) or key == ord(' '):
            os_controller.toggle_active()
            print(f"[STATUS] Controller active state: {os_controller.is_enabled}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
