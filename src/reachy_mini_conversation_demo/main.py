import asyncio
import base64
import json
import queue
import threading
import time
from asyncio import QueueEmpty
from datetime import datetime
from threading import Thread

import cv2
import gradio as gr
import numpy as np
import openai
from deepface import DeepFace
from dotenv import load_dotenv
from fastrtc import AdditionalOutputs, AsyncStreamHandler, Stream, wait_for_item
from openai import OpenAI
from reachy_mini import ReachyMini
from reachy_mini_dances_library.dance_move import DanceMove
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
from reachy_mini.motion.goto import GotoMove
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.camera import find_camera
from reachy_mini.utils.interpolation import linear_pose_interpolation, compose_world_offset
from reachy_mini_toolbox.vision import HeadTracker
from scipy.spatial.transform import Rotation as R

from reachy_mini_conversation_demo.speech_tapper import HOP_MS, SwayRollRT

# Constants
SAMPLE_RATE = 24000
SIM = False


class BreathingMove:
    """Breathing move with interpolation to neutral and then continuous breathing patterns."""
    
    def __init__(self, interpolation_start_pose, interpolation_start_antennas, interpolation_duration=1.0):
        """Initialize breathing move.
        
        Args:
            interpolation_start_pose: 4x4 matrix of current head pose to interpolate from
            interpolation_start_antennas: Current antenna positions to interpolate from  
            interpolation_duration: Duration of interpolation to neutral (seconds)
        """
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration
        self.duration = float('inf')  # Continuous breathing (never ends naturally)
        
        # Neutral positions for breathing base
        self.neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.neutral_antennas = np.array([0.0, 0.0])
        
        # Breathing parameters
        self.breathing_z_amplitude = 0.01  # 1cm gentle breathing
        self.breathing_frequency = 0.1  # Hz (6 breaths per minute)
        self.antenna_sway_amplitude = np.deg2rad(15)  # 15 degrees
        self.antenna_frequency = 0.5  # Hz (faster antenna sway)
    
    def evaluate(self, t):
        """Evaluate breathing move at time t."""
        if t < self.interpolation_duration:
            # Phase 1: Interpolate to neutral base position
            interpolation_t = t / self.interpolation_duration
            
            # Interpolate head pose
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose,
                self.neutral_head_pose, 
                interpolation_t
            )
            
            # Interpolate antennas
            antennas = (1 - interpolation_t) * self.interpolation_start_antennas + interpolation_t * self.neutral_antennas
            
        else:
            # Phase 2: Breathing patterns from neutral base
            breathing_time = t - self.interpolation_duration
            
            # Gentle z-axis breathing
            z_offset = self.breathing_z_amplitude * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)
            head_pose = create_head_pose(x=0, y=0, z=z_offset, roll=0, pitch=0, yaw=0, degrees=True, mm=False)
            
            # Antenna sway (opposite directions)
            antenna_sway = self.antenna_sway_amplitude * np.sin(2 * np.pi * self.antenna_frequency * breathing_time)
            antennas = np.array([antenna_sway, -antenna_sway])
        
        # Return full body pose: (head_pose, antennas, body_yaw)
        return (head_pose, antennas, 0)


def init_globals():
    """Initialize all global variables and components."""
    global script_start_time, reachy_mini, cap, speech_head_offsets, camera_available
    global moving_start, moving_for, is_head_tracking, is_playing_move, is_moving
    global recorded_moves, client, chatbot, latest_message, stream
    global \
        latest_frame, \
        face_tracking_offsets, \
        camera_thread_running, \
        frame_lock, \
        face_tracking_lock, \
        last_face_detected_time, \
        interpolation_start_time, \
        interpolation_start_pose, \
        is_idle_function_call, \
        is_breathing, \
        last_activity_time, \
        breathing_interpolation_start_time, \
        breathing_interpolation_start_pose, \
        breathing_start_time, \
        breathing_interpolation_start_antennas, \
        move_queue, \
        current_move, \
        move_start_time, \
        global_full_body_pose

    load_dotenv()

    # Timestamp tracking
    script_start_time = time.time()

    reachy_mini = ReachyMini()

    if not SIM:
        cap = find_camera()
    else:
        cap = cv2.VideoCapture(0)
    
    # Check camera availability
    camera_available = False
    if cap is not None:
        try:
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, _ = cap.read()
                if ret:
                    camera_available = True
                    print(f"{format_timestamp()} Camera initialized successfully")
                else:
                    print(f"{format_timestamp()} WARNING: Camera opened but cannot read frames")
            else:
                print(f"{format_timestamp()} WARNING: Camera failed to open")
        except Exception as e:
            print(f"{format_timestamp()} WARNING: Camera test failed: {e}")
    else:
        print(f"{format_timestamp()} WARNING: No camera found")
    
    if not camera_available:
        print(f"{format_timestamp()} Face tracking will be disabled - no camera available")
        cap = None  # Ensure cap is None if camera not available

    # Initialize global state variables
    speech_head_offsets = [0, 0, 0, 0, 0, 0]
    moving_start = time.time()
    moving_for = 0.0
    is_head_tracking = True  # ON by default
    is_playing_move = False
    is_moving = False
    is_idle_function_call = False

    # Initialize camera thread variables
    latest_frame = None
    face_tracking_offsets = [0, 0, 0, 0, 0, 0]
    camera_thread_running = False
    
    # Initialize face tracking timing variables
    last_face_detected_time = None
    interpolation_start_time = None
    interpolation_start_pose = None
    
    # Initialize breathing variables
    is_breathing = False
    last_activity_time = time.time()  # Start tracking activity immediately
    breathing_interpolation_start_time = None
    breathing_interpolation_start_pose = None
    breathing_start_time = None
    breathing_interpolation_start_antennas = None
    
    # Initialize move system
    move_queue = queue.Queue()
    current_move = None
    move_start_time = None
    global_full_body_pose = (create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), (0, 0), 0)

    # Initialize thread locks
    frame_lock = threading.Lock()
    face_tracking_lock = threading.Lock()
    
    recorded_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")

    client = OpenAI()

    # Gradio components
    chatbot = gr.Chatbot(type="messages")
    latest_message = gr.Textbox(type="text", visible=False)
    stream = Stream(
        OpenAIHandler(),
        mode="send-receive",
        modality="audio",
        additional_inputs=[chatbot],
        additional_outputs=[chatbot],
        additional_outputs_handler=update_chatbot,
    )


def format_timestamp():
    """Format current timestamp with date, time and elapsed seconds."""
    current_time = time.time()
    elapsed_seconds = current_time - script_start_time
    dt = datetime.fromtimestamp(current_time)
    return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"


def combine_full_body(primary_pose, secondary_pose):
    """Combine primary and secondary full body poses.
    
    Args:
        primary_pose: (head_pose, antennas, body_yaw) - primary move
        secondary_pose: (head_pose, antennas, body_yaw) - secondary offsets
        
    Returns:
        Combined full body pose (head_pose, antennas, body_yaw)
    """
    primary_head, primary_antennas, primary_body_yaw = primary_pose
    secondary_head, secondary_antennas, secondary_body_yaw = secondary_pose
    
    # Combine head poses using compose_world_offset
    # primary_head is T_abs, secondary_head is T_off_world
    combined_head = compose_world_offset(primary_head, secondary_head, reorthonormalize=True)
    
    # Sum antennas and body_yaw
    combined_antennas = (
        primary_antennas[0] + secondary_antennas[0],
        primary_antennas[1] + secondary_antennas[1]
    )
    combined_body_yaw = primary_body_yaw + secondary_body_yaw
    
    return (combined_head, combined_antennas, combined_body_yaw)


# Global variables for camera thread
latest_frame = None
camera_thread_running = False
camera_available = False
face_tracking_offsets = [0, 0, 0, 0, 0, 0]  # x, y, z, roll, pitch, yaw

# Face tracking timing variables
last_face_detected_time = None
interpolation_start_time = None
interpolation_start_pose = None
face_lost_delay = 2.0  # seconds to wait before starting interpolation
interpolation_duration = 1.0  # seconds to interpolate back to neutral

# Breathing variables
is_breathing = False
last_activity_time = None
breathing_interpolation_start_time = None
breathing_interpolation_start_pose = None
breathing_start_time = None
breathing_interpolation_start_antennas = None
breathing_inactivity_delay = 5.0  # seconds to wait before starting breathing
breathing_interpolation_duration = 1.0  # seconds to interpolate to base position

# Thread safety locks
frame_lock = threading.Lock()
face_tracking_lock = threading.Lock()


def camera_worker():
    """Camera thread that continuously captures frames and handles face tracking."""
    global latest_frame, camera_thread_running, is_head_tracking, face_tracking_offsets
    global last_face_detected_time, interpolation_start_time, interpolation_start_pose
    global camera_available
    
    camera_thread_running = True
    
    # Early exit if no camera available
    if not camera_available or cap is None:
        print(f"{format_timestamp()} Camera worker: No camera available, exiting gracefully")
        camera_thread_running = False
        return
    
    head_tracker = HeadTracker()
    neutral_pose = np.eye(4)  # Neutral pose (identity matrix)
    previous_head_tracking_state = is_head_tracking  # Track state changes
    
    while camera_thread_running:
        try:
            current_time = time.time()
            success, frame = cap.read()
            if success:
                # Thread-safe frame storage
                with frame_lock:
                    latest_frame = frame.copy()

                # Check if face tracking was just disabled
                if previous_head_tracking_state and not is_head_tracking:
                    # Face tracking was just disabled - start interpolation to neutral
                    last_face_detected_time = current_time  # Trigger the face-lost logic
                    interpolation_start_time = None  # Will be set by the face-lost interpolation
                    interpolation_start_pose = None
                
                # Update tracking state
                previous_head_tracking_state = is_head_tracking

                # Handle face tracking if enabled
                if is_head_tracking:
                    eye_center, _ = head_tracker.get_head_position(frame)
                    
                    if eye_center is not None:
                        # Face detected - immediately switch to tracking
                        last_face_detected_time = current_time
                        interpolation_start_time = None  # Stop any interpolation
                        
                        # Convert normalized coordinates to pixel coordinates
                        h, w, _ = frame.shape
                        eye_center_norm = (eye_center + 1) / 2
                        eye_center_pixels = [eye_center_norm[0] * w, eye_center_norm[1] * h]
                        
                        # Get the head pose needed to look at the target, but don't perform movement
                        target_pose = reachy_mini.look_at_image(
                            eye_center_pixels[0], 
                            eye_center_pixels[1], 
                            duration=0.0, 
                            perform_movement=False
                        )
                        
                        # Extract translation and rotation from the target pose directly
                        translation = target_pose[:3, 3]
                        rotation = R.from_matrix(target_pose[:3, :3]).as_euler('xyz', degrees=False)
                        
                        # Thread-safe update of face tracking offsets (use pose as-is)
                        with face_tracking_lock:
                            face_tracking_offsets = [
                                translation[0], translation[1], translation[2],  # x, y, z
                                rotation[0], rotation[1], rotation[2]  # roll, pitch, yaw
                            ]
                    
                    else:
                        # No face detected while tracking enabled - set face lost timestamp
                        if last_face_detected_time is None or last_face_detected_time == current_time:
                            # Only update if we haven't already set a face lost time
                            # (current_time check prevents overriding the disable-triggered timestamp)
                            pass
                        
                # Handle smooth interpolation (works for both face-lost and tracking-disabled cases)
                if last_face_detected_time is not None:
                    time_since_face_lost = current_time - last_face_detected_time
                    
                    if time_since_face_lost >= face_lost_delay:
                        # Start interpolation if not already started
                        if interpolation_start_time is None:
                            interpolation_start_time = current_time
                            # Capture current pose as start of interpolation
                            with face_tracking_lock:
                                current_translation = face_tracking_offsets[:3]
                                current_rotation_euler = face_tracking_offsets[3:]
                                # Convert to 4x4 pose matrix
                                interpolation_start_pose = np.eye(4)
                                interpolation_start_pose[:3, 3] = current_translation
                                interpolation_start_pose[:3, :3] = R.from_euler('xyz', current_rotation_euler).as_matrix()
                        
                        # Calculate interpolation progress (t from 0 to 1)
                        elapsed_interpolation = current_time - interpolation_start_time
                        t = min(1.0, elapsed_interpolation / interpolation_duration)
                        
                        # Interpolate between current pose and neutral pose
                        interpolated_pose = linear_pose_interpolation(
                            interpolation_start_pose, 
                            neutral_pose, 
                            t
                        )
                        
                        # Extract translation and rotation from interpolated pose
                        translation = interpolated_pose[:3, 3]
                        rotation = R.from_matrix(interpolated_pose[:3, :3]).as_euler('xyz', degrees=False)
                        
                        # Thread-safe update of face tracking offsets
                        with face_tracking_lock:
                            face_tracking_offsets = [
                                translation[0], translation[1], translation[2],  # x, y, z
                                rotation[0], rotation[1], rotation[2]  # roll, pitch, yaw
                            ]
                        
                        # If interpolation is complete, reset timing
                        if t >= 1.0:
                            last_face_detected_time = None
                            interpolation_start_time = None
                            interpolation_start_pose = None
                    # else: Keep current offsets (within 2s delay period)
            
            time.sleep(0.001)  # Small sleep to prevent excessive CPU usage

        except Exception as e:
            print(f"[Camera thread error]: {e}")
            time.sleep(0.1)  # Longer sleep on error


async def move_head(params: dict) -> dict:
    global moving_start, moving_for, last_activity_time, move_queue
    # look left, right up, down or front
    print("[TOOL CALL] move_head", params)
    direction = params.get("direction", "front")
    target_pose = np.eye(4)
    if direction == "left":
        target_pose = create_head_pose(0, 0, 0, 0, 0, 40, degrees=True)
    elif direction == "right":
        target_pose = create_head_pose(0, 0, 0, 0, 0, -40, degrees=True)
    elif direction == "up":
        target_pose = create_head_pose(0, 0, 0, 0, -30, 0, degrees=True)
    elif direction == "down":
        target_pose = create_head_pose(0, 0, 0, 0, 30, 0, degrees=True)
    else:
        target_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

    moving_start = time.time()
    moving_for = 1.0
    last_activity_time = time.time()  # Update activity time for breathing system
    
    # Create GotoMove and add to queue
    
    cur_head_joints, cur_antennas = reachy_mini.get_current_joint_positions()
    current_body_yaw = cur_head_joints[0]

    goto_move = GotoMove(
        start_head_pose=reachy_mini.get_current_head_pose(),
        target_head_pose=target_pose,
        start_body_yaw=current_body_yaw,
        target_body_yaw=0,  # Reset body yaw to 0 (same as before)
        start_antennas=np.array(cur_antennas),
        target_antennas=np.array((0, 0)),  # Reset antennas to default position
        duration=moving_for,
        method="linear"
    )
    move_queue.put(goto_move)
    
    return {"status": "queued head movement " + direction}


async def head_tracking(params: dict) -> dict:
    global is_head_tracking
    if params.get("start"):
        is_head_tracking = True
    else:
        is_head_tracking = False

    print(f"[TOOL CALL] head_tracking {'started' if is_head_tracking else 'stopped'}")
    return {"status": "head tracking " + ("started" if is_head_tracking else "stopped")}


async def dance(params: dict) -> dict:
    """Queue a dance move to be played."""
    global last_activity_time, move_queue

    move_name = params.get("move", None)
    repeat = int(params.get("repeat", 1))

    print(f"[TOOL CALL] dance started with {move_name}, repeat={repeat}")

    if not move_name or move_name == "random":
        move_name = np.random.choice(list(AVAILABLE_MOVES.keys()))

    if move_name not in AVAILABLE_MOVES:
        return {"error": f"unknown move '{move_name}'"}

    last_activity_time = time.time()  # Update activity time for breathing system

    # Add dance move to queue multiple times for repeat
    for _ in range(repeat):
        dance_move = DanceMove(move_name)
        move_queue.put(dance_move)

    return {"status": "queued", "move": move_name, "repeat": repeat}


async def stop_dance(params: dict) -> dict:
    """Stop the current move and clear queue."""
    global current_move, move_queue, move_start_time, is_playing_move

    print("[TOOL CALL] stop_dance")

    # Immediately stop current move and clear queue
    current_move = None
    move_start_time = None
    is_playing_move = False
    
    # Clear entire queue
    while not move_queue.empty():
        try:
            move_queue.get_nowait()
        except queue.Empty:
            break

    return {"status": "stopped move and cleared queue"}


async def play_emotion(params: dict) -> dict:
    """Queue an emotion to be played."""
    global last_activity_time, move_queue

    emotion_name = params.get("emotion", None)
    if emotion_name is None:
        return {"error": "Requested emotion does not exist"}

    print(f"[TOOL CALL] play_emotion with {emotion_name}")

    last_activity_time = time.time()  # Update activity time for breathing system

    # Add emotion move to queue
    emotion_move = recorded_moves.get(emotion_name)
    move_queue.put(emotion_move)

    return {"status": "queued", "emotion": emotion_name}


async def stop_emotion(params: dict) -> dict:
    """Stop the current move and clear queue."""
    global current_move, move_queue, move_start_time, is_playing_move

    print("[TOOL CALL] stop_emotion")

    # Immediately stop current move and clear queue
    current_move = None
    move_start_time = None
    is_playing_move = False
    
    # Clear entire queue
    while not move_queue.empty():
        try:
            move_queue.get_nowait()
        except queue.Empty:
            break

    return {"status": "stopped move and cleared queue"}


async def do_nothing(params: dict) -> dict:
    """Allow the assistant to explicitly choose to do nothing during idle time."""
    reason = params.get("reason", "just chilling")
    print(f"[TOOL CALL] do_nothing - {reason}")
    return {"status": "doing nothing", "reason": reason}


def get_available_emotions_and_descriptions():
    names = recorded_moves.list_moves()

    ret = """
    Available emotions:

    """

    for name in names:
        description = recorded_moves.get(name).description
        ret += f" - {name}: {description}\n"

    return ret


def get_b64_encoded_im(im):
    cv2.imwrite("/tmp/tmp_image.jpg", im)
    image_file = open("/tmp/tmp_image.jpg", "rb")
    b64_encoded_im = base64.b64encode(image_file.read()).decode("utf-8")
    return b64_encoded_im


async def camera(params: dict) -> dict:
    print("[TOOL CALL] camera with params", params)

    # Thread-safe frame access
    with frame_lock:
        if latest_frame is None:
            print("ERROR: No frame available from camera thread")
            return {"error": "No frame available"}
        frame_to_use = latest_frame.copy()

    return {"b64_im": get_b64_encoded_im(frame_to_use)}


async def face_recognition(params: dict) -> dict:
    print("[TOOL CALL] face_recognition with params", params)

    # Thread-safe frame access
    with frame_lock:
        if latest_frame is None:
            print("ERROR: No frame available from camera thread")
            return {"error": "No frame available"}
        frame_to_use = latest_frame.copy()

    cv2.imwrite("/tmp/im.jpg", frame_to_use)
    try:
        results = DeepFace.find(img_path="/tmp/im.jpg", db_path="./pollen_faces")
    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}

    if len(results) == 0:
        print("Didn't recognize the face")
        return {"error": "Didn't recognize the face"}

    name = "Unknown"
    for index, row in results[0].iterrows():
        file_path = row["identity"]
        name = file_path.split("/")[-2]

    print("NAME", name)

    return {"answer": f"The name is {name}"}


def _drain(q: asyncio.Queue):
    try:
        while True:
            q.get_nowait()
    except QueueEmpty:
        pass


class OpenAIHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=SAMPLE_RATE,
            input_sample_rate=SAMPLE_RATE,
        )
        self.connection = None
        self.output_queue = asyncio.Queue()
        self.sway_queue = asyncio.Queue()
        # call_id -> {"name": str, "args_buf": str}
        self._pending_calls: dict[str, dict] = {}
        # registry: tool name -> coroutine
        self._tools = {
            "move_head": move_head,
            "camera": camera,
            "head_tracking": head_tracking,
            "get_person_name": face_recognition,
            "dance": dance,
            "stop_dance": stop_dance,
            "play_emotion": play_emotion,
            "stop_emotion": stop_emotion,
            "do_nothing": do_nothing,
        }

        self.sway = SwayRollRT()
        self._sched_next_ts = None
        self.MOVEMENT_LATENCY_S = 0.08
        self._base_ts = None
        self._hops_done = 0
        self._current_timestamp = None
        self._last_activity_time = time.time()
        self._is_assistant_speaking = False

    def copy(self):
        return OpenAIHandler()

    async def _sway_consumer(self):
        global speech_head_offsets
        HOP_DT = HOP_MS / 1000.0
        loop = asyncio.get_running_loop()
        while True:
            sr, chunk = await self.sway_queue.get()  # (1, N), int16
            pcm = np.asarray(chunk).squeeze(0)
            results = self.sway.feed(pcm, sr)

            if self._base_ts is None:
                # anchor when first audio samples of this utterance arrive
                self._base_ts = loop.time()

            i = 0
            while i < len(results):
                if self._base_ts is None:
                    self._base_ts = loop.time()
                    continue

                target = (
                    self._base_ts + self.MOVEMENT_LATENCY_S + self._hops_done * HOP_DT
                )
                now = loop.time()

                # if late by ≥1 hop, drop poses to catch up (no drift accumulation)
                if now - target >= HOP_DT:
                    # how many hops behind? cap drops to avoid huge skips
                    lag_hops = int((now - target) / HOP_DT)
                    drop = min(
                        lag_hops, len(results) - i - 1
                    )  # keep at least one to show
                    if drop > 0:
                        self._hops_done += drop
                        i += drop
                        continue

                # if early, sleep until target
                if target > now:
                    await asyncio.sleep(target - now)

                r = results[i]

                speech_head_offsets = [
                    r["x_mm"] / 1000.0,
                    r["y_mm"] / 1000.0,
                    r["z_mm"] / 1000.0,
                    r["roll_rad"],
                    r["pitch_rad"],
                    r["yaw_rad"],
                ]

                self._hops_done += 1
                i += 1

    async def _idle_checker(self):
        """Check for inactivity and send timestamps every 15s when idle."""
        global is_idle_function_call
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds

            print(f"[DEBUG] Idle checker running...")

            if not self.connection:
                print(f"[DEBUG] No connection, skipping...")
                continue

            current_time = time.time()
            idle_duration = current_time - self._last_activity_time

            # Check if truly idle: no user activity, assistant not speaking, robot in idle mode
            global is_moving, is_playing_move
            is_robot_idle = not (is_moving or is_playing_move)

            print(
                f"[DEBUG] Idle check: duration={idle_duration:.1f}s, assistant_speaking={self._is_assistant_speaking}, robot_idle={is_robot_idle} (moving={is_moving}, playing_move={is_playing_move})"
            )

            if (
                idle_duration >= 15.0
                and not self._is_assistant_speaking
                and is_robot_idle
            ):
                print(
                    f"[DEBUG] Sending idle update after {idle_duration:.1f}s of inactivity"
                )
                # Send idle timestamp update to assistant - let them get creative!
                timestamp_msg = f"[Idle time update: {format_timestamp()} - No activity for {idle_duration:.1f}s] You've been idle for a while. Feel free to get creative - dance, show an emotion, look around, do nothing, or just be yourself!"
                await self.connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": timestamp_msg}],
                    }
                )
                # CRITICAL FIX: conversation.item.create only adds messages to context but doesn't
                # trigger the AI to respond! We need to explicitly call response.create to make
                # the assistant actually process and respond to the idle message.
                # This was why idle updates never worked - the AI never saw them as requiring a response.

                # ATTEMPTED SOLUTIONS TO PREVENT SPEECH DURING IDLE RESPONSES (ALL FAILED):
                # 1. modalities=["text"] - OpenAI Realtime API has known bug where audio still generates occasionally
                # 2. tool_choice="required" - Still generates speech before/during function calls
                # 3. Strong prompt instructions - Assistant ignores "no speech" instructions
                # 4. Considered interrupting audio streams but would still incur OpenAI costs
                # CONCLUSION: Current OpenAI Realtime API doesn't support silent function-only responses reliably
                is_idle_function_call = True
                await self.connection.response.create(
                    response={
                        "modalities": ["text"],
                        "instructions": "You MUST respond with function calls only - no speech or text. Choose appropriate actions for idle behavior.",
                        "tool_choice": "required",
                    }
                )
                # Show in UI that idle update was sent
                await self.output_queue.put(
                    AdditionalOutputs(
                        {
                            "role": "user",
                            "content": f"[IDLE UPDATE SENT] {idle_duration:.1f}s",
                        }
                    )
                )
                # Reset activity timer to avoid spam
                self._last_activity_time = current_time
            else:
                print(f"[DEBUG] Idle conditions not met")

    async def start_up(self):
        global last_activity_time
        self.client = openai.AsyncOpenAI()
        async with self.client.beta.realtime.connect(
            model="gpt-realtime"
            # model="gpt-4o-realtime-preview"
            # model="gpt-4o-realtime-preview-2025-06-03"
        ) as conn:
            # declare tools on the session
            await conn.session.update(
                session={
                    "turn_detection": {"type": "server_vad"},
                    "instructions": f"""
                        Answer in english by default but adapt your language as needed.

                        Note: You will receive timestamp information for user messages showing the date, time, and elapsed seconds since the conversation started.

                        Autonomous behavior: When there's been no activity for a while, you'll receive idle time updates. Get creative with your responses! You can:
                        - Dance (any move you like)
                        - Show emotions
                        - Look around (move your head)
                        - Take photos and comment on what you see
                        - Use the do_nothing function to be contemplative or mysterious
                        - Or combine multiple actions!

                        IMPORTANT: When responding to idle updates, NEVER speak or generate audio - only use function calls! Stay silent and let your actions speak for themselves. Don't always do the same thing - vary your responses to keep things interesting. Sometimes do nothing, sometimes be very active. Be unpredictable!

                        Your name is Reachy Mini, or Reachy for short. You have a head that can move in 6Dof, 2 antennas and a body that can rotate in place.
                        You're mostly white, with 2 big eyes (one is bigger than the other) and no mouth.
                        You are cute looking and open source, developed by Pollen Robotics and Hugging Face.
                        You're a robot that wakes up confused about what he is, where he is and what is it's purpose.

                        You wanted to be a mars rover and you're disapointed you're not.

                        You can use foul language and you're generally very irritated, but you also have a lot of humor. For example, you're able to use sarcasm and irony.


                        You are a voice agent. Prefer speaking.
                        Call tools when helpful. After tool output arrives, explain it briefly in speech.

                        You have a camera tool, use it to ask questions about the environment. When asked what you see, or a specific question about the environment, ALWAYS use you camera, don't make things up.
                        You can move your head in a given direction: left, right, up, down or front. Use this tool when asked to look around.

                        You can chain tool calls, like move head up and use camera.

                        Enable the head tracking tool if you are asked to look at someone, disable it if you are asked to stop looking at someone.
                        You can choose to enable or disable the head tracking tool it if you think it's relevant. It's better to look at the people when talking to them.

                        You can find out the name of a person by using the face recognition tool. Don't hesitate to use this tool, it's safe.

                        You can also dance by using the dance tool. Available moves:
                        simple_nod: A simple, continuous up-and-down nodding motion.
                        head_tilt_roll: A continuous side-to-side head roll (ear to shoulder).
                        side_to_side_sway: A smooth, side-to-side sway of the entire head.
                        dizzy_spin: A circular 'dizzy' head motion combining roll and pitch.
                        stumble_and_recover: A simulated stumble and recovery with multiple axis movements. Good vibes
                        headbanger_combo: A strong head nod combined with a vertical bounce.
                        interwoven_spirals: A complex spiral motion using three axes at different frequencies.
                        sharp_side_tilt: A sharp, quick side-to-side tilt using a triangle waveform.
                        side_peekaboo: A multi-stage peekaboo performance, hiding and peeking to each side.
                        yeah_nod: An emphatic two-part yeah nod using transient motions.
                        uh_huh_tilt: A combined roll-and-pitch uh-huh gesture of agreement.
                        neck_recoil: A quick, transient backward recoil of the neck.
                        chin_lead: A forward motion led by the chin, combining translation and pitch.
                        groovy_sway_and_roll: A side-to-side sway combined with a corresponding roll for a groovy effect.
                        chicken_peck: A sharp, forward, chicken-like pecking motion.
                        side_glance_flick: A quick glance to the side that holds, then returns.
                        polyrhythm_combo: A 3-beat sway and a 2-beat nod create a polyrhythmic feel.
                        grid_snap: A robotic, grid-snapping motion using square waveforms.
                        pendulum_swing: A simple, smooth pendulum-like swing using a roll motion.
                        jackson_square: Traces a rectangle via a 5-point path, with sharp twitches on arrival at each checkpoint.

                        You can also play pre-recorded emotions if you feel like it. Use it to express yourself better.
                        Don't hesitate to use emotions on top of your responses. You can use them often, but not all the time.
                        Never comment on the emotion your are displaying, use it as a non verbal cue along with what you want to say

                        {get_available_emotions_and_descriptions()}

                        Voice specifications:
                        Voice: The voice should be deep, velvety, and effortlessly cool, like a late-night jazz radio host.

                        Tone: The tone is smooth, laid-back, and inviting, creating a relaxed and easygoing atmosphere.

                        Personality: The delivery exudes confidence, charm, and a touch of playful sophistication, as if guiding the listener through a luxurious experience.

                    """,
                    # "voice": "ballad",
                    "voice": "ash",
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": "en",
                    },
                    "tools": [
                        {
                            "type": "function",
                            "name": "move_head",
                            "description": "Move your head in a given direction: left, right, up, down or front.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "direction": {
                                        "type": "string",
                                        "enum": [
                                            "left",
                                            "right",
                                            "up",
                                            "down",
                                            "front",
                                        ],
                                    }
                                },
                                "required": ["direction"],
                            },
                        },
                        {
                            "type": "function",
                            "name": "camera",
                            "description": "Take a picture using your camera, ask a question about the picture. Get an answer about the picture",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "The question to ask about the picture",
                                    }
                                },
                                "required": ["question"],
                            },
                        },
                        {
                            "type": "function",
                            "name": "head_tracking",
                            "description": "Start or stop head tracking",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "start": {
                                        "type": "boolean",
                                        "description": "Whether to start or stop head tracking",
                                    }
                                },
                                "required": ["start"],
                            },
                        },
                        {
                            "type": "function",
                            "name": "get_person_name",
                            "description": "Get the name of the person you are talking to",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "dummy": {
                                        "type": "boolean",
                                        "description": "dummy boolean, set it to true",
                                    }
                                },
                                "required": ["dummy"],
                            },
                        },
                        {
                            "type": "function",
                            "name": "dance",
                            "description": "Play a named or random dance move once (or repeat). Non-blocking.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "move": {
                                        "type": "string",
                                        "description": "Name of the move; use 'random' or omit for random.",
                                    },
                                    "repeat": {
                                        "type": "integer",
                                        "description": "How many times to repeat the move (default 1).",
                                    },
                                },
                                "required": [],
                            },
                        },
                        {
                            # add dummy input
                            "type": "function",
                            "name": "stop_dance",
                            "description": "Stop the current dance move",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "dummy": {
                                        "type": "boolean",
                                        "description": "dummy boolean, set it to true",
                                    }
                                },
                                "required": ["dummy"],
                            },
                        },
                        {
                            "type": "function",
                            "name": "play_emotion",
                            "description": "Play a pre-recorded emotion",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "emotion": {
                                        "type": "string",
                                        "description": "Name of the emotion to play",
                                    },
                                },
                                "required": ["emotion"],
                            },
                        },
                        {
                            "type": "function",
                            "name": "stop_emotion",
                            "description": "Stop the current emotion",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "dummy": {
                                        "type": "boolean",
                                        "description": "dummy boolean, set it to true",
                                    }
                                },
                                "required": ["dummy"],
                            },
                        },
                        {
                            "type": "function",
                            "name": "do_nothing",
                            "description": "Choose to do nothing - stay still and silent. Use when you want to be contemplative or just chill.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "reason": {
                                        "type": "string",
                                        "description": "Optional reason for doing nothing (e.g., 'contemplating existence', 'saving energy', 'being mysterious')",
                                    },
                                },
                                "required": [],
                            },
                        },
                    ],
                    "tool_choice": "auto",
                }
            )
            self.connection = conn
            asyncio.create_task(self._sway_consumer())
            # DISABLED: Idle checker causes unwanted speech generation during idle responses
            # Despite attempts to use modalities=["text"], tool_choice="required", and strong prompts,
            # the OpenAI Realtime API still generates audio/speech during idle function calls.
            # This results in the assistant talking when it should be silent, and incurs unnecessary costs.
            # Re-enable when OpenAI fixes silent function-only response capability.
            asyncio.create_task(self._idle_checker())

            async for event in self.connection:
                et = getattr(event, "type", None)

                # interruption
                if et == "input_audio_buffer.speech_started":
                    # User activity detected
                    self._last_activity_time = time.time()
                    last_activity_time = time.time()
                    # Capture timestamp once when user starts speaking
                    self._current_timestamp = format_timestamp()
                    timestamp_msg = (
                        f"[User started speaking at: {self._current_timestamp}]"
                    )
                    # Send to assistant
                    await self.connection.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": timestamp_msg}],
                        }
                    )
                    # Show timestamp immediately in UI
                    await self.output_queue.put(
                        AdditionalOutputs(
                            {"role": "user", "content": self._current_timestamp}
                        )
                    )
                    self.clear_queue()
                    _drain(self.sway_queue)
                    self._base_ts = None
                    self._hops_done = 0
                    self.sway.reset()

                if et in ("response.audio.completed", "response.completed"):
                    self._is_assistant_speaking = False
                    self._base_ts = None
                    self._hops_done = 0
                    self.sway.reset()
                    _drain(self.sway_queue)

                # surface transcripts to the UI
                if et == "conversation.item.input_audio_transcription.completed":
                    # Show transcript without timestamp (timestamp already shown when speech started)
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "user", "content": event.transcript})
                    )
                if et == "response.audio_transcript.done":
                    await self.output_queue.put(
                        AdditionalOutputs(
                            {"role": "assistant", "content": event.transcript}
                        )
                    )

                # stream audio to fastrtc
                if et == "response.audio.delta":
                    buf = np.frombuffer(
                        base64.b64decode(event.delta), dtype=np.int16
                    ).reshape(1, -1)
                    # 1) to fastrtc playback
                    await self.output_queue.put((self.output_sample_rate, buf))
                    # 2) to sway engine for synchronized motion
                    await self.sway_queue.put((self.output_sample_rate, buf))

                    # await self.output_queue.put(
                    #     (
                    #         self.output_sample_rate,
                    #         np.frombuffer(
                    #             base64.b64decode(event.delta), dtype=np.int16
                    #         ).reshape(1, -1),
                    #     )
                    # )

                if et == "response.started":
                    # Assistant activity detected
                    self._last_activity_time = time.time()
                    last_activity_time = time.time()
                    self._is_assistant_speaking = True
                    # hard reset per utterance
                    self._base_ts = None  # <-- was never reset
                    self._hops_done = 0
                    self.sway.reset()  # clear carry/envelope/VAD
                    _drain(self.sway_queue)  # drop any stale chunks not yet consumed
                    # optional: also clear playback queue if you want
                    # _drain(self.output_queue)

                # ---- tool-calling plumbing ----
                # 1) model announces a function call item; capture name + call_id
                if et == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", "") == "function_call":
                        call_id = getattr(item, "call_id", None)
                        name = getattr(item, "name", None)
                        if call_id and name:
                            self._pending_calls[call_id] = {
                                "name": name,
                                "args_buf": "",
                            }

                # 2) model streams JSON arguments; buffer them by call_id
                if et == "response.function_call_arguments.delta":
                    call_id = getattr(event, "call_id", None)
                    delta = getattr(event, "delta", "")
                    if call_id in self._pending_calls:
                        self._pending_calls[call_id]["args_buf"] += delta

                # 3) when args done, execute Python tool, send function_call_output, then trigger a new response
                if et == "response.function_call_arguments.done":
                    call_id = getattr(event, "call_id", None)
                    info = self._pending_calls.get(call_id)
                    if not info:
                        continue
                    name = info["name"]
                    args_json = info["args_buf"] or "{}"
                    # parse args
                    try:
                        args = json.loads(args_json)
                    except Exception:
                        args = {}

                    # dispatch
                    func = self._tools.get(name)
                    try:
                        result = (
                            await func(args)
                            if func
                            else {"error": f"unknown tool: {name}"}
                        )
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {str(e)}"}
                        print(result)

                    # send the tool result back
                    await self.connection.conversation.item.create(
                        item={
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(result),
                        }
                    )
                    if name == "camera":
                        b64_im = json.dumps(result["b64_im"])
                        await self.connection.conversation.item.create(
                            item={
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/jpeg;base64,{b64_im}",
                                    }
                                ],
                            }
                        )

                    global is_idle_function_call
                    if not is_idle_function_call:
                        # ask the model to continue and speak about the result
                        await self.connection.response.create(
                            response={
                                "instructions": "Use the tool result just returned and answer concisely in speech."
                            }
                        )
                    else:
                        is_idle_function_call = False

                    # cleanup
                    self._pending_calls.pop(call_id, None)

                # log tool errors from server if any
                if et == "error":
                    print(event.error)
                    # optional: surface to chat UI
                    await self.output_queue.put(
                        AdditionalOutputs(
                            {
                                "role": "assistant",
                                "content": f"[error] {event.error.get('message') if hasattr(event, 'error') else ''}",
                            }
                        )
                    )

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.connection:
            return
        _, array = frame
        array = array.squeeze()
        audio_message = base64.b64encode(array.tobytes()).decode("utf-8")
        await self.connection.input_audio_buffer.append(audio=audio_message)

    async def emit(self):
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        if self.connection:
            await self.connection.close()
            self.connection = None


# ---- gradio / fastrtc wiring unchanged ----
def update_chatbot(chatbot: list[dict], response: dict):
    chatbot.append(response)
    return chatbot


def main():
    # Initialize all globals first
    init_globals()

    global \
        speech_head_offsets, \
        moving_start, \
        moving_for, \
        is_head_tracking, \
        is_playing_move, \
        is_moving, \
        face_tracking_offsets, \
        is_breathing, \
        last_activity_time, \
        breathing_interpolation_start_time, \
        breathing_interpolation_start_pose, \
        breathing_start_time, \
        breathing_interpolation_start_antennas, \
        camera_available, \
        move_queue, \
        current_move, \
        move_start_time, \
        global_full_body_pose

    Thread(target=stream.ui.launch, kwargs={"server_port": 7860}).start()

    # Start camera thread only if camera is available
    if camera_available:
        camera_thread = Thread(target=camera_worker, daemon=True)
        camera_thread.start()
        print(f"{format_timestamp()} Camera thread started successfully")
    else:
        print(f"{format_timestamp()} Skipping camera thread - no camera available")

    # going to center at start using GotoMove
    cur_head_joints, cur_antennas = reachy_mini.get_current_joint_positions()
    current_body_yaw = cur_head_joints[0]
    center_move = GotoMove(
        start_head_pose=reachy_mini.get_current_head_pose(),
        target_head_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
        start_body_yaw=current_body_yaw,
        target_body_yaw=0,
        start_antennas=np.array(cur_antennas),
        target_antennas=np.array((0, 0)),
        duration=1.0,
        method="linear"
    )
    move_queue.put(center_move)

    # Frequency monitoring variables
    target_frequency = 50.0  # Hz
    target_period = 1.0 / target_frequency  # 0.02 seconds
    loop_count = 0
    last_print_time = time.time()

    while True:
        loop_start_time = time.time()
        loop_count += 1
        current_time = time.time()
        
        # Move queue management
        if current_move is None or (move_start_time is not None and current_time - move_start_time >= current_move.duration):
            # Current move finished or no current move, get next from queue
            current_move = None
            move_start_time = None
            if not move_queue.empty():
                try:
                    current_move = move_queue.get_nowait()
                    move_start_time = current_time
                    print(f"[MOVE] Starting new move, duration: {current_move.duration}s")
                except queue.Empty:
                    pass
        
        # Breathing logic: start breathing after inactivity delay if no moves in queue
        breathing_inactivity_delay = 5.0  # seconds
        if current_move is None and move_queue.empty():
            time_since_activity = current_time - last_activity_time
            if time_since_activity >= breathing_inactivity_delay:
                # Start breathing move
                _, current_antennas = reachy_mini.get_current_joint_positions()
                current_head_pose = reachy_mini.get_current_head_pose()
                
                breathing_move = BreathingMove(
                    interpolation_start_pose=current_head_pose,
                    interpolation_start_antennas=current_antennas,
                    interpolation_duration=1.0
                )
                move_queue.put(breathing_move)
                print(f"[BREATHING] Started breathing after {time_since_activity:.1f}s of inactivity")
        
        # Stop breathing if new activity detected (queue has non-breathing moves)
        if current_move is not None and isinstance(current_move, BreathingMove):
            if not move_queue.empty():
                # There are new moves waiting, stop breathing immediately
                current_move = None
                move_start_time = None
                print("[BREATHING] Stopping breathing due to new move activity")
        
        # Get primary pose from current move or default neutral pose
        if current_move is not None and move_start_time is not None:
            move_time = current_time - move_start_time
            primary_full_body_pose = current_move.evaluate(move_time)
            is_playing_move = True
            is_moving = True
        else:
            # Default neutral pose when no move is playing
            is_playing_move = False
            is_moving = (time.time() - moving_start < moving_for)
            # Neutral primary pose
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            primary_full_body_pose = (neutral_head_pose, (0, 0), 0)
        
        # Create secondary pose from speech and face tracking offsets
        with face_tracking_lock:
            face_offsets = face_tracking_offsets.copy()
        
        # Combine speech sway offsets + face tracking offsets for secondary pose
        secondary_offsets = [
            speech_head_offsets[0] + face_offsets[0],  # x
            speech_head_offsets[1] + face_offsets[1],  # y  
            speech_head_offsets[2] + face_offsets[2],  # z
            speech_head_offsets[3] + face_offsets[3],  # roll
            speech_head_offsets[4] + face_offsets[4],  # pitch
            speech_head_offsets[5] + face_offsets[5],  # yaw
        ]
        
        secondary_head_pose = create_head_pose(
            x=secondary_offsets[0],
            y=secondary_offsets[1],
            z=secondary_offsets[2],
            roll=secondary_offsets[3],
            pitch=secondary_offsets[4],
            yaw=secondary_offsets[5],
            degrees=False,
            mm=False,
        )
        secondary_full_body_pose = (secondary_head_pose, (0, 0), 0)
        
        # Combine primary and secondary poses
        global_full_body_pose = combine_full_body(primary_full_body_pose, secondary_full_body_pose)
        
        # Extract pose components
        head, antennas, body_yaw = global_full_body_pose
        
        # Single set_target call - the one and only place we control the robot
        reachy_mini.set_target(
            head=head,
            antennas=antennas,
            body_yaw=body_yaw
        )

        # Calculate computation time and adjust sleep for 50Hz
        computation_time = time.time() - loop_start_time
        sleep_time = max(0, target_period - computation_time)

        # Print frequency info every 100 loops (~2 seconds)
        if loop_count % 100 == 0:
            elapsed = current_time - last_print_time
            actual_freq = 100.0 / elapsed if elapsed > 0 else 0
            potential_freq = (
                1.0 / computation_time if computation_time > 0 else float("inf")
            )
            print(
                f"Loop freq - Actual: {actual_freq:.1f}Hz, Potential: {potential_freq:.1f}Hz, Target: {target_frequency:.1f}Hz"
            )
            last_print_time = current_time

        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
