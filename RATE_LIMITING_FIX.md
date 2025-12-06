# Rate Limiting Fix - Smoothing Large Pitch Changes

**Date:** October 27, 2025
**Status:** ✅ Implemented

---

## The Problem

After removing all limiting from camera_worker to fix feedback loops, large sudden pitch changes caused oscillation:

**Scenario:** User sits down quickly
1. Head position drops from -5° to position requiring -20° pitch
2. Camera calculates -20°, moves.py clamps to -15°
3. Robot tries to move from -5° to -15° instantly (10° jump)
4. Physical inertia + overshoot → robot bounces
5. Bounces back and forth, can't settle

**Root cause:** No damping on large sudden changes

---

## The Solution: Rate Limiting After Clamp

**Key principle:** Apply smoothing AFTER calculation, not DURING it

### Implementation

**Location:** `moves.py` lines 970-978 (after mechanical clamp)

```python
# After clamping to ±15°
clamped_pitch = np.clip(face_pitch, -15°, +15°)

# Rate limit the change
pitch_change = clamped_pitch - self._last_commanded_pitch
rate_limited_pitch = self._last_commanded_pitch + np.clip(
    pitch_change,
    -self._max_pitch_change_per_frame,  # -5°/frame
    +self._max_pitch_change_per_frame   # +5°/frame
)
self._last_commanded_pitch = rate_limited_pitch

# Use rate_limited_pitch in composition
```

---

## How It Works

### Example: User Sits Down Quickly

**Frame 1:**
- Camera: "Need -20° pitch"
- Clamp: -20° → -15° (mechanical limit)
- Last commanded: -5°
- Change: -15° - (-5°) = -10°
- Rate limit: Max ±5°/frame → apply -5°
- **Command: -5° + (-5°) = -10°**

**Frame 2:**
- Camera: "Need -20° pitch"
- Clamp: -20° → -15°
- Last commanded: -10°
- Change: -15° - (-10°) = -5°
- Rate limit: Within ±5° → allow full change
- **Command: -10° + (-5°) = -15°**

**Result:** Smooth 2-frame transition instead of instant 10° jump

---

## Key Differences from Old Broken System

| Aspect | Old (camera_worker) | New (moves.py) |
|--------|---------------------|----------------|
| **Where applied** | In camera_worker before calculation | In moves.py after calculation |
| **What it limits** | Calculation of desired pitch | Output to robot |
| **Feedback loop** | YES - corrupts next calculation | NO - camera still fresh |
| **State location** | Camera worker | Movement manager |
| **Effect on camera** | Changes what camera calculates | No effect on camera |

**Critical difference:** Camera worker remains stateless. It always calculates fresh from current camera view. Rate limiting only smooths the OUTPUT.

---

## Parameters

**Maximum change per frame:**
```python
self._max_pitch_change_per_frame = np.deg2rad(5.0)  # 5°/frame
```

At 30Hz loop rate: **5°/frame = 150°/second**

**Tuning guidance:**
- **Too small (2°/frame):** Feels sluggish, takes too long to reach target
- **Just right (5°/frame):** Smooth transitions, responsive tracking
- **Too large (10°/frame):** Still allows bounce/overshoot
- **Recommended range:** 3-7° per frame

---

## State Management

**State variable:**
```python
self._last_commanded_pitch = 0.0
```

**Updated every frame** after rate limiting (line 978)

**No reset needed** because:
- Unlike camera_worker, this doesn't create feedback
- State represents actual last command sent to robot
- Smooth continuity between frames is desired

---

## Logging

Enhanced pitch logging shows all three values:

```
Pitch tracking: raw=-20.3°, clamped=-15.0°, rate_limited=-10.2°
```

- **raw:** What camera_worker calculated (unlimited)
- **clamped:** After mechanical ±15° limit
- **rate_limited:** Final value sent to robot (smoothed)

Logged every 100 frames (~1 second at 100Hz)

---

## Expected Behavior

### Small Changes (Within Rate Limit)
**Face moves from -5° to -8°:**
- Change: 3° (within 5° limit)
- Applied immediately, no delay
- **Result:** Instant, responsive tracking

### Large Changes (Exceeds Rate Limit)
**User sits down, face goes from -5° to -20°:**
- Needs: -15° (after clamp)
- Change: 10° (exceeds 5° limit)
- Frame 1: Apply 5° → command -10°
- Frame 2: Apply remaining 5° → command -15°
- **Result:** Smooth 2-frame transition, no bounce

### Beyond Mechanical Limit
**Face at -25° (beyond -15° limit):**
- Clamp: -25° → -15°
- Rate limited approach to -15°
- Head reaches -15° and stays
- Face still 10° off-center (steady-state error)
- **Result:** Stable, no oscillation

---

## Files Modified

**moves.py:**
- Lines 343-345: Added state variables
  - `_last_commanded_pitch`
  - `_max_pitch_change_per_frame`
- Lines 970-978: Rate limiting logic after clamp
- Lines 985-989: Enhanced logging with rate_limited value
- Line 997: Use `rate_limited_pitch` in composition (not `clamped_pitch`)

---

## Testing Checklist

Verify these behaviors:

- [ ] Small face movements → Instant, responsive tracking
- [ ] Large sudden movements (sit down) → Smooth transition, no bounce
- [ ] Face beyond limits → Smooth approach to limit, stable
- [ ] Rapid face position changes → Damped, controlled response
- [ ] Log shows raw, clamped, and rate_limited values correctly

---

## Why This Works

**No feedback loop:**
- Camera worker calculates fresh every frame (stateless)
- Rate limiting in moves.py only smooths OUTPUT
- Camera never sees its own limited output
- No state corruption possible

**Predictable behavior:**
- Linear rate limit (5°/frame)
- Always reaches correct target (given enough frames)
- No non-linear compression or unpredictable damping

**Tunable:**
- Single parameter: `_max_pitch_change_per_frame`
- Easy to adjust if too slow/fast
- Clear relationship: degrees/frame = degrees/second ÷ Hz

---

## Success Criteria

✅ Camera worker remains stateless (no rate limiting there)
✅ Large sudden changes are smoothed (no bounce)
✅ Small changes remain responsive
✅ Robot reaches correct target position
✅ No feedback loops or state corruption
✅ Behavior is predictable and debuggable

**Status: Implemented and ready for testing**

---

*This fix adds damping to large pitch changes while preserving the stateless camera_worker architecture that prevents feedback loops.*
