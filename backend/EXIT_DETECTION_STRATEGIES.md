# Exit Detection Strategies

## Current Approach: Timeout-Based
**How it works**: Mark exit after X seconds of absence
- ✅ Simple and reliable
- ❌ May mark exit too early if person is still in frame but temporarily not detected
- ❌ Doesn't distinguish between "left the area" vs "temporary occlusion"

## Alternative Strategies

### 1. **Frame-Based Exit (Consistent with Entry)**
**How it works**: Count consecutive frames of absence (mirrors entry logic)
- ✅ Consistent with entry detection logic
- ✅ More predictable behavior
- ❌ May be too sensitive (temporary occlusions trigger exit)

**Example**:
```python
# If employee not detected for 15 consecutive frames → mark exit
# At 5 FPS: 15 frames = 3 seconds
```

### 2. **Movement-Based Exit Detection**
**How it works**: Track bounding box movement toward frame edges
- ✅ More accurate for actual exits
- ✅ Distinguishes "leaving" from "temporary occlusion"
- ❌ Requires tracking bbox position over time
- ❌ More complex implementation

**Logic**:
- Track bbox position history
- If bbox moves toward edge (left/right/top) and then disappears → mark exit
- If bbox disappears from center → wait longer (might be occlusion)

### 3. **Zone-Based Exit Detection**
**How it works**: Define exit zones (edges of frame)
- ✅ Intentional exit detection
- ✅ Can distinguish entry/exit zones
- ❌ Requires zone configuration
- ❌ May not work if camera angle changes

**Zones**:
- **Exit zones**: Left edge (10%), Right edge (10%), Top edge (10%)
- **Center zone**: Middle 80% of frame
- If person detected in exit zone → prepare for exit
- If person disappears from exit zone → mark exit faster

### 4. **Size-Based Exit Detection**
**How it works**: Track bounding box size (person moving away)
- ✅ Natural exit detection (person gets smaller as they leave)
- ✅ Works well for distance-based exits
- ❌ May not work if person just moves closer/farther without leaving

**Logic**:
- Track bbox area over time
- If bbox shrinks significantly (e.g., 50% smaller) → person moving away
- If bbox shrinks AND disappears → mark exit

### 5. **Hybrid Approach (Recommended)**
**How it works**: Combine multiple signals for robust detection
- ✅ Most reliable
- ✅ Handles edge cases better
- ❌ More complex

**Combination**:
1. **Primary**: Frame-based absence counting (consistent with entry)
2. **Secondary**: Movement toward edges (if available)
3. **Fallback**: Timeout (safety net)

### 6. **Confidence-Based Exit**
**How it works**: Track detection confidence over time
- ✅ Handles partial occlusions
- ❌ May be too complex

**Logic**:
- If confidence drops gradually → person leaving
- If confidence drops suddenly → temporary occlusion (don't mark exit)

## Recommended Implementation

I recommend implementing a **Hybrid Approach** with these priorities:

1. **Frame-based absence** (primary): Count consecutive frames of absence
2. **Movement tracking** (secondary): If bbox was moving toward edge, reduce threshold
3. **Timeout fallback** (safety): Maximum time before forced exit

This gives you:
- **Fast exit** when person clearly leaves (movement + absence)
- **Reliable exit** for normal cases (frame counting)
- **Safe exit** for edge cases (timeout)

