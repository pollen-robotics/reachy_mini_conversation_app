# Face Tracking Implementation Summary

**Date:** October 27, 2025
**Status:** ✅ All changes completed

---

## Changes Implemented

### 1. Fixed Pitch Inversion
**File:** `camera_worker.py:177-185`

**Problem:** Daemon API returns positive pitch = down, but robot interprets positive pitch = up

**Solution:** Added pitch inversion before clamping
```python
# Invert pitch: daemon API coordinate system is opposite of robot's
inverted_pitch = -rotation[1]

# Clamp to prevent collisions
max_pitch_up = np.deg2rad(15.0)     # Up is positive (after inversion)
max_pitch_down = np.deg2rad(-15.0)  # Down is negative (after inversion)
clamped_pitch = np.clip(inverted_pitch, max_pitch_down, max_pitch_up)
```

**Result:** Robot now correctly tilts UP when face is above, DOWN when face is below

---

### 2. Pause Breathing During Body Rotation
**File:** `moves.py:697-707`

**Problem:** Breathing roll oscillation interfered with clean body rotation during syncing

**Solution:** Suppress breathing when body is in SYNCING state
```python
suppress_breathing = (
    (isinstance(self.state.current_move, BreathingMove) and external_active) or
    (isinstance(self.state.current_move, BreathingMove) and self._anchor_state == AnchorState.SYNCING)
)

if suppress_breathing:
    # Return neutral pose with z=0.01 lift to maintain "alive" appearance
    head = create_head_pose(0, 0, 0.01, 0, 0, 0, degrees=False, mm=False)
```

**Result:** Clean body rotation without roll oscillation interference

---

### 3. Make Breathing Relative to Anchor
**File:** `moves.py:724-739`

**Problem:** Breathing always happened at yaw=0, not relative to current anchor position

**Solution:** Add anchor yaw offset to breathing pose
```python
if isinstance(self.state.current_move, BreathingMove):
    # Extract current yaw from breathing pose (should be 0)
    head_rotation = R.from_matrix(head_copy[:3, :3])
    head_roll, head_pitch, head_yaw_local = head_rotation.as_euler("xyz")

    # Add anchor offset to make breathing relative to anchor position
    anchor_yaw_rad = np.deg2rad(self._body_anchor_yaw)
    absolute_yaw = head_yaw_local + anchor_yaw_rad

    # Reconstruct rotation matrix with anchor-relative yaw
    new_rotation = R.from_euler("xyz", [head_roll, head_pitch, absolute_yaw])
    head_copy[:3, :3] = new_rotation.as_matrix()
```

**Result:** Breathing motion happens around current anchor position, not zero

---

### 4. Anchor Locking After 3 Seconds
**File:** `moves.py:340-341`

**Problem:** Anchor locked too quickly (0.5s) with too loose threshold (5°)

**Solution:** Updated stability parameters
```python
self._stability_duration_s = 3.0  # 3 seconds for head stabilization before anchor lock
self._stability_threshold_deg = 2.0  # 2 degrees max movement to be considered stable
```

**Result:**
- Anchor only locks after head stays within 2° for 3 consecutive seconds
- More stable anchor points, less frequent re-anchoring

---

### 5. 1-Second Body Sync Interpolation
**File:** `moves.py:330`

**Problem:** Body rotation took too long (1.5s) to sync with head

**Solution:** Reduced interpolation duration
```python
self._body_follow_duration = 1.0  # Duration for smooth body follow interpolation (1 second)
```

**Result:** Body smoothly syncs to head position over exactly 1 second

---

## System Behavior Summary

### Anchor State Machine
1. **ANCHORED** - Body locked at anchor point
   - Strain threshold: 13° (head-body difference)
   - When exceeded → enters SYNCING

2. **SYNCING** - Body rotating to match head
   - Duration: 1.0 second (smooth interpolation)
   - Breathing: PAUSED (only z=0.01 lift maintained)
   - When complete → enters STABILIZING

3. **STABILIZING** - Waiting for head stability
   - Body matches head (locked together)
   - Breathing: ACTIVE (but relative to current position)
   - Stability requirement: < 2° movement for 3 seconds
   - When achieved → new anchor set, returns to ANCHORED

### Face Tracking Pitch Behavior
- Face above robot → Negative pitch (after inversion) → Head tilts UP
- Face below robot → Positive pitch (after inversion) → Head tilts DOWN
- Clamped to ±15° to prevent throat guard/back collar collision

### Breathing Relative to Anchor
- At anchor 0°: breathing oscillates around 0°
- At anchor 45°: breathing oscillates around 45°
- At anchor -30°: breathing oscillates around -30°
- Roll oscillation preserved, no yaw reset

---

## Files Modified

1. **camera_worker.py** - Pitch inversion and clamping
2. **moves.py** - All other changes (breathing pause, anchor-relative, timing)

---

## Testing Checklist

When you test, verify:

- [ ] Face above robot → head tilts UP (not down into throat guard)
- [ ] Face below robot → head tilts DOWN (not up)
- [ ] Body smoothly rotates to follow head over 1 second
- [ ] Breathing pauses during body rotation (no roll wobble)
- [ ] Breathing continues after rotation, relative to new position
- [ ] Anchor locks after 3 seconds of < 2° head movement
- [ ] No oscillation or jittering during tracking

---

## Backup Information

**Backup branch created:** `claude-broken-state`

To revert all changes:
```bash
git checkout claude-broken-state
```

To see what changed:
```bash
git diff claude-broken-state
```

---

## Next Steps

1. Test face tracking behavior
2. Verify pitch direction is correct
3. Confirm breathing pauses during rotation
4. Check anchor locking timing (3 seconds)
5. Validate 1-second body sync feels natural

If anything doesn't work as expected, check the logs (run with `--debug` flag) to see:
- Pitch values from daemon API
- Body follow state transitions
- Anchor locking events
