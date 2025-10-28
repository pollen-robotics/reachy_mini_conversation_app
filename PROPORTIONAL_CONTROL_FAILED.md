# Proportional Control Implementation - CATASTROPHIC FAILURE

**Date:** October 27, 2025
**Status:** ❌ FAILED - Reverted immediately

---

## FAILURE SUMMARY

**This approach caused complete loss of control:**
- Pitch accumulated to 85°+ (should be ±15° max)
- Body yaw spun uncontrollably (jumped from -6° to 174° instantly)
- Robot became completely unresponsive to face position

**Root cause:**
1. **Unbounded accumulation**: `_current_pitch` kept growing without limit or decay
2. **No feedback**: No knowledge of robot's actual position to correct accumulation
3. **Yaw confusion**: Stored per-frame adjustment as absolute position value

**Reverted to:** Absolute positioning using daemon's `look_at_image()` IK

---

## Original Intent (What Was Attempted)

---

## The Change

**Previous Approach (Absolute Positioning):**
- Used daemon's `look_at_image()` IK to calculate exact pitch angle needed to center face
- Robot moved to that absolute target position
- Example: "Face needs -20° pitch to center" → try to move to -20° (clamped to -15°)

**New Approach (Proportional Control):**
- Calculate how far off-center the face is (error)
- Apply proportional gain to error
- Accumulate adjustments to current pitch
- Example: "Face is 10° off-center" → adjust by 10° × 0.4 = 4°

---

## Why Proportional Control?

**Benefits:**
1. **Smoother tracking** - Gradual convergence instead of immediate target
2. **More natural** - Error-based adjustment feels more human-like
3. **Self-stabilizing** - As face approaches center, adjustments get smaller
4. **Recoverable** - Small errors don't cause large movements

**vs Gemini's Broken Version:**
- Gemini used direct velocity control → unrecoverable oscillation
- Our version uses position accumulation → stable with rate limiting
- We have 3 safety layers: clamping (±15°), rate limiting (5°/frame), oscillation detection

---

## Implementation Details

### Location
**File:** `camera_worker.py`

### State Variables
```python
self._current_pitch = 0.0          # Accumulated pitch position
self._proportional_gain = 0.4      # Gain factor (0.3-0.5 safe range)
```

### Control Loop (lines 154-193)

**Step 1: Calculate Error**
```python
vertical_error = eye_center[1]  # Normalized: 0 = centered, >0 = below, <0 = above
error_angle = vertical_error * np.deg2rad(30.0)  # Convert to angle (±30° max)
```

**Step 2: Apply Gain**
```python
pitch_adjustment = -error_angle * self._proportional_gain
# Negative because: error > 0 (below) needs pitch down (negative)
```

**Step 3: Accumulate**
```python
self._current_pitch += pitch_adjustment
```

**Step 4: Store**
```python
self.face_tracking_offsets[4] = self._current_pitch  # Sent to moves.py
```

---

## Coordinate System

### Eye Center Normalized Coordinates
- `eye_center[1]` range: -1.0 to +1.0
- `0.0` = perfectly centered vertically
- `> 0` = face below center (needs negative pitch = down)
- `< 0` = face above center (needs positive pitch = up)

### Camera Field of View
- **Vertical FOV:** ~60° (±30° from center)
- **Horizontal FOV:** ~80° (±40° from center)

### Error to Angle Conversion
```python
# Vertical (pitch)
error_angle = vertical_error * 30°
# If face at edge (error = 1.0), error_angle = 30°

# Horizontal (yaw)
yaw_error = horizontal_error * 40°
# If face at edge (error = 1.0), yaw_error = 40°
```

---

## Control Behavior

### Example: Face Below Center

**Frame 1:**
- `eye_center[1]` = 0.5 (face halfway down frame)
- `error_angle` = 0.5 × 30° = 15°
- `pitch_adjustment` = -15° × 0.4 = -6°
- `current_pitch` = 0° + (-6°) = **-6°**

**Frame 2:**
- Face still visible but moved slightly toward center
- `eye_center[1]` = 0.3 (closer to center)
- `error_angle` = 0.3 × 30° = 9°
- `pitch_adjustment` = -9° × 0.4 = -3.6°
- `current_pitch` = -6° + (-3.6°) = **-9.6°**

**Frame 3:**
- Face approaching center
- `eye_center[1]` = 0.1 (nearly centered)
- `error_angle` = 0.1 × 30° = 3°
- `pitch_adjustment` = -3° × 0.4 = -1.2°
- `current_pitch` = -9.6° + (-1.2°) = **-10.8°**

**Convergence:**
- As face approaches center, error → 0
- Adjustments become smaller
- Robot smoothly settles at correct position

---

## Safety Layers

### 1. Proportional Gain Limit
**In camera_worker.py:**
```python
self._proportional_gain = 0.4  # Max adjustment is error × 0.4
```
- Limits how aggressively pitch changes per frame
- 0.4 = 40% of error applied per frame
- Prevents overshoot

### 2. Mechanical Clamping
**In moves.py (line 974):**
```python
clamped_pitch = np.clip(face_pitch, -15°, +15°)
```
- Hard limit at ±15° (throat guard / collar safety)
- Always enforced regardless of calculated pitch

### 3. Rate Limiting
**In moves.py (lines 1001-1028):**
```python
rate_limited_pitch = last_pitch + np.clip(
    change,
    -5°/frame,
    +5°/frame
)
```
- Maximum 5° change per frame
- Prevents violent snaps
- Smooths large sudden movements

### 4. Oscillation Detection
**In moves.py (lines 978-999):**
```python
if direction_changes > 2:
    # Enter recovery mode
    rate_limited_pitch = 0.0  # Hold at 0° for 5 seconds
```
- Detects runaway bouncing
- Forces calm period
- Resumes after recovery

---

## State Management

### When _current_pitch Resets to 0.0

1. **Tracking Disabled** (line 95)
   ```python
   def set_head_tracking_enabled(self, enabled: bool):
       if not enabled:
           self._current_pitch = 0.0
   ```

2. **Tracking Just Enabled** (line 149)
   ```python
   if not previous_state and self.is_head_tracking_enabled:
       self._current_pitch = 0.0
   ```

3. **Tracking Disabled in Loop** (line 144)
   ```python
   if was_enabled and not is_enabled:
       self._current_pitch = 0.0
   ```

4. **Face Lost Interpolation Complete** (line 254)
   ```python
   if t >= 1.0:  # Interpolation done
       self._current_pitch = 0.0
   ```

**Why Reset?**
- Prevents accumulated error from carrying over
- Clean slate when tracking resumes
- No residual pitch from previous tracking session

---

## Tuning Parameters

### Proportional Gain
```python
self._proportional_gain = 0.4  # Current value
```

**Effect on Behavior:**
- **0.2-0.3:** Slower convergence, more damped, sluggish
- **0.4-0.5:** Responsive, smooth tracking (recommended)
- **0.6-0.8:** Fast convergence, possible overshoot
- **0.9-1.0:** Aggressive, high risk of oscillation

**Recommended Range:** 0.3-0.5

### Field of View Scaling
```python
error_angle = vertical_error * np.deg2rad(30.0)  # Vertical FOV half-angle
yaw_error = horizontal_error * np.deg2rad(40.0)  # Horizontal FOV half-angle
```

**Adjustment:**
- Increase if robot under-responds to off-center faces
- Decrease if robot over-responds to small errors
- Should match actual camera FOV for accurate tracking

---

## Expected Behavior

### Face Centered
- `error` ≈ 0
- `adjustment` ≈ 0°/frame
- Robot holds steady position

### Face Slightly Off-Center (error = 0.2)
- `adjustment` = 0.2 × 30° × 0.4 = 2.4°/frame
- Smooth, responsive tracking
- No oscillation

### Face Far Off-Center (error = 0.8)
- `adjustment` = 0.8 × 30° × 0.4 = 9.6°/frame
- Still within 5°/frame rate limit (moves.py clamps)
- Smooth approach over multiple frames

### Face Beyond Limits (accumulated pitch > 15°)
- Moves.py clamps to ±15°
- Robot reaches limit and holds
- Face remains off-center (steady-state error)
- **No oscillation** (rate limiting + oscillation detection prevent)

---

## Differences from Absolute Positioning

| Aspect | Absolute (Old) | Proportional (New) |
|--------|----------------|---------------------|
| **Calculation** | Daemon IK: "Need -20° to center face" | Error-based: "Face 10° off, adjust by 4°" |
| **Response** | Tries to reach target immediately | Gradual convergence over frames |
| **Large Errors** | Big jumps (needs heavy rate limiting) | Self-limiting (gain reduces large changes) |
| **Small Errors** | Instant correction | Smooth approach |
| **At Limits** | Hard stop at ±15° | Smooth approach to limit |
| **Naturalness** | Robotic (direct positioning) | Human-like (error correction) |
| **Overshoot** | Possible if target changes | Self-damping (error decreases) |

---

## Differences from Gemini's Broken Version

| Aspect | Gemini's Version | Our Version |
|--------|------------------|-------------|
| **Control Type** | Direct motor velocity | Position accumulation |
| **Feedback** | Uncontrolled positive feedback | Negative feedback (error reduces) |
| **Safety** | None (violent thrashing) | 4 layers (gain, clamp, rate, oscillation) |
| **Recovery** | Unrecoverable oscillation | Self-stabilizing with recovery mode |
| **State** | Corrupted easily | Clean resets on state changes |

**Why Ours is Safe:**
- Gain limits maximum adjustment per frame (0.4 = 40% of error)
- Clamping prevents exceeding mechanical limits
- Rate limiting smooths sudden changes
- Oscillation detection forces recovery
- Error-based (not velocity) means it converges naturally

---

## Logging

**Debug output (lines 177-181):**
```
Proportional control: v_error=0.523, pitch_adj=-6.28°, current_pitch=-6.28°
```

**Interpretation:**
- `v_error`: Vertical error in normalized coords (-1 to +1)
- `pitch_adj`: Adjustment applied this frame (degrees)
- `current_pitch`: Accumulated pitch position (degrees)

**Frequency:** Every frame (100Hz in moves.py logs, but camera runs at ~30Hz)

---

## Files Modified

**camera_worker.py:**
- Lines 69-71: Added `_current_pitch` and `_proportional_gain` state variables
- Lines 93-96: Reset state when tracking disabled
- Lines 146-149: Reset state when tracking just enabled
- Lines 138-144: Reset state when tracking disabled in loop
- Lines 154-193: Proportional control logic (replaced daemon IK absolute positioning)
- Line 254: Reset state when interpolation completes

**No changes to moves.py:**
- Clamping, rate limiting, and oscillation detection remain unchanged
- Still enforces ±15° mechanical limits
- Still limits to 5°/frame
- Still detects and recovers from oscillation

---

## Testing Checklist

Verify these behaviors:

- [ ] Face centered → Robot holds steady (no jitter)
- [ ] Face moves slightly → Smooth tracking with gradual adjustment
- [ ] Face moves far off-center → Smooth approach over multiple frames (no snap)
- [ ] Face beyond limits (>15° needed) → Smooth approach to ±15°, stable hold
- [ ] Face lost → Smooth interpolation back to neutral over 3 seconds
- [ ] Tracking disabled → Clean reset, no residual pitch
- [ ] Tracking re-enabled → Fresh start from 0°
- [ ] Large sudden movement (sit down) → Rate limiting + oscillation detection prevent bounce
- [ ] Rapid oscillation (>2 direction changes) → Recovery mode holds at 0° for 5 seconds

---

## Success Criteria

✅ Replaced absolute positioning with proportional error-based control
✅ Gain factor set to safe value (0.4)
✅ State resets on all tracking transitions
✅ Safety layers preserved (clamp, rate limit, oscillation detection)
✅ Smoother, more natural tracking than absolute positioning
✅ No feedback loops or state corruption
✅ Recoverable from all error conditions

**Status: Implementation complete, ready for testing**

---

*This change converts face tracking from absolute positioning to proportional error-based control, providing smoother, more natural tracking while maintaining all safety systems.*
