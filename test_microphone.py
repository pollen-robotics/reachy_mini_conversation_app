#!/usr/bin/env python3
"""Quick microphone test script to diagnose audio input issues."""

import time
import numpy as np
from reachy_mini import ReachyMini

print("=" * 60)
print("MICROPHONE TEST for Reachy Mini")
print("=" * 60)

# Connect to robot
print("\n1. Connecting to robot...")
robot = ReachyMini()
print("   ✓ Connected")

# Get audio info
print("\n2. Audio system information:")
sample_rate = robot.media.get_audio_samplerate()
print(f"   Sample rate: {sample_rate} Hz")

# Start recording
print("\n3. Starting recording...")
robot.media.start_recording()
print("   ✓ Recording started")
print("   Waiting 1 second for pipeline to initialize...")
time.sleep(1.0)

# Capture some frames
print("\n4. Capturing audio frames (5 seconds)...")
print("   ⚠️  SPEAK NOW to test the microphone!")
print()

frames_captured = 0
total_samples = 0
max_amplitude = 0
sum_rms = 0.0

start_time = time.time()
while time.time() - start_time < 5.0:
    frame = robot.media.get_audio_sample()
    if frame is not None:
        frames_captured += 1
        
        # Convert to mono
        mono = frame.T[0]
        total_samples += len(mono)
        
        # Calculate RMS and max
        rms = np.sqrt(np.mean(mono.astype(np.float32)**2))
        max_val = np.max(np.abs(mono))
        
        sum_rms += rms
        max_amplitude = max(max_amplitude, max_val)
        
        # Show a simple level meter
        level = int(max_val * 50)  # Scale to 50 chars
        bar = "█" * level
        print(f"\r   Level: [{bar:<50}] RMS: {rms:.4f}, Max: {max_val:.4f}", end="", flush=True)
        
    time.sleep(0.01)

print()  # New line after progress
print("\n5. Results:")
print(f"   Frames captured: {frames_captured}")
print(f"   Total samples: {total_samples}")
print(f"   Average RMS: {sum_rms / max(frames_captured, 1):.4f}")
print(f"   Max amplitude: {max_amplitude:.4f}")

print("\n6. Diagnosis:")
if max_amplitude < 0.001:
    print("   ❌ PROBLEM: No audio detected (silent)")
    print("   Possible causes:")
    print("      - Microphone is muted or not connected")
    print("      - Wrong microphone device selected")
    print("      - Microphone permissions not granted")
    print("      - Hardware issue with robot's microphone")
elif max_amplitude < 0.01:
    print("   ⚠️  WARNING: Very low audio levels")
    print("      - Try speaking louder")
    print("      - Check microphone sensitivity settings")
elif max_amplitude < 0.5:
    print("   ✓ GOOD: Audio detected at normal levels")
    print("      - Microphone is working correctly")
else:
    print("   ⚠️  WARNING: Very loud audio (possible clipping)")
    print("      - Audio might be distorted")
    print("      - Try speaking more quietly")

# Stop recording
print("\n7. Stopping recording...")
robot.media.stop_recording()
robot.client.disconnect()
print("   ✓ Test complete")

print("\n" + "=" * 60)
print("If the microphone test shows no audio, check:")
print("  1. System audio settings (System Preferences > Sound > Input)")
print("  2. Which microphone device is set as default")
print("  3. Microphone permissions for Python/Terminal")
print("  4. Robot microphone hardware connection")
print("=" * 60)

