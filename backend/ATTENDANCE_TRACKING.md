# Attendance Tracking System

## Overview

The attendance tracking system automatically marks employees as "present" (entry) and "exited" based on live face detection. The system tracks consecutive detections and absence periods to determine when to mark attendance.

## Database Schema

### Attendance Table

The `attendance` table stores employee entry and exit records:

```sql
CREATE TABLE attendance (
    id UUID PRIMARY KEY,
    employee_id UUID REFERENCES employees(id),
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);
```

**Key Features:**
- One entry per employee per day (if exit_time is NULL, employee is still present)
- Automatic timestamp updates
- Indexed for fast queries by employee, date, and entry/exit times

## Configuration

The attendance tracking behavior is controlled by configuration in `live_detection.py`:

```python
ATTENDANCE_CONFIG = {
    'MIN_CONSECUTIVE_DETECTIONS': 5,  # Frames to mark entry
    'EXIT_STRATEGY': 'hybrid',        # Exit detection strategy
    'EXIT_FRAME_COUNT': 15,           # Frames of absence to mark exit (frame-based)
    'EXIT_TIMEOUT_SECONDS': 30,        # Seconds of absence (timeout fallback)
    'EXIT_ZONE_THRESHOLD': 0.1,        # Exit zone size (10% of frame edges)
    'EXIT_MOVEMENT_FRAMES': 3,         # Frames to track movement
    'COOLDOWN_SECONDS': 5              # Cooldown between actions
}
```

### Configuration Parameters

1. **MIN_CONSECUTIVE_DETECTIONS** (default: 5)
   - Number of consecutive frames where an employee must be detected before marking entry
   - At 5 FPS (200ms per frame), 5 frames = ~1 second
   - **Recommendation**: 5-10 frames (1-2 seconds) for reliable detection

2. **EXIT_STRATEGY** (default: 'hybrid')
   - **'timeout'**: Simple timeout-based exit (original approach)
   - **'frame_based'**: Count consecutive frames of absence (consistent with entry)
   - **'hybrid'**: Combines frame-based + movement tracking + timeout fallback (recommended)
   - **Recommendation**: Use 'hybrid' for best results

3. **EXIT_FRAME_COUNT** (default: 15)
   - Consecutive frames of absence before marking exit (for frame-based/hybrid)
   - At 5 FPS, 15 frames = ~3 seconds
   - **Recommendation**: 10-20 frames (2-4 seconds)

4. **EXIT_TIMEOUT_SECONDS** (default: 30)
   - Seconds of absence before marking exit (for timeout strategy or hybrid fallback)
   - **Recommendation**: 30-60 seconds

5. **EXIT_ZONE_THRESHOLD** (default: 0.1)
   - Percentage of frame edges considered "exit zones" (0.1 = 10%)
   - Used in hybrid strategy to detect intentional exits
   - **Recommendation**: 0.1-0.15 (10-15% of frame)

6. **EXIT_MOVEMENT_FRAMES** (default: 3)
   - Number of recent frames to check for movement toward exit zone
   - **Recommendation**: 3-5 frames

7. **COOLDOWN_SECONDS** (default: 5)
   - Minimum time between attendance actions (prevents duplicate entries/exits)
   - **Recommendation**: 5-10 seconds

## How It Works

### Entry Detection

1. **Detection Phase**: Employee face is detected and recognized in video frames
2. **Counting Phase**: System counts consecutive frames where employee is detected
3. **Entry Marking**: When count reaches `MIN_CONSECUTIVE_DETECTIONS`, entry is marked in database
4. **State Tracking**: System tracks that entry has been marked (prevents duplicate entries)

### Exit Detection (Multiple Strategies)

The exit detection strategy is configurable via `EXIT_STRATEGY`:

#### Strategy 1: Timeout-Based (Simple)
1. **Absence Detection**: Employee is no longer detected in frames
2. **Timeout Phase**: System tracks time since last detection
3. **Exit Marking**: When absence exceeds `EXIT_TIMEOUT_SECONDS`, exit is marked
4. **Pros**: Simple, reliable
5. **Cons**: May mark exit too early if person temporarily not detected

#### Strategy 2: Frame-Based (Consistent with Entry)
1. **Absence Detection**: Employee is no longer detected in frames
2. **Frame Counting**: System counts consecutive frames of absence
3. **Exit Marking**: When count reaches `EXIT_FRAME_COUNT`, exit is marked
4. **Pros**: Consistent with entry logic, predictable
5. **Cons**: May be sensitive to temporary occlusions

#### Strategy 3: Hybrid (Recommended) ⭐
1. **Movement Tracking**: Tracks if bounding box was moving toward frame edges (exit zones)
2. **Fast Exit**: If person was moving toward exit zone, uses faster threshold (half of `EXIT_FRAME_COUNT`)
3. **Normal Exit**: If person disappears from center, uses normal `EXIT_FRAME_COUNT` threshold
4. **Fallback**: Timeout as safety net if frame-based doesn't trigger
5. **Pros**: Most accurate, handles edge cases, distinguishes "leaving" from "occlusion"
6. **Cons**: More complex

**Exit Zone Detection**:
- Defines edges of frame (10% by default) as "exit zones"
- If person's bounding box center is near edges → likely leaving
- If person disappears from center → likely temporary occlusion (wait longer)

### Flow Diagram

#### Entry Detection (All Strategies)
```
Frame 1-4: Employee detected → Count: 1, 2, 3, 4 (no action)
Frame 5: Employee detected → Count: 5 → MARK ENTRY ✓
Frame 6+: Employee detected → Entry already marked (no action)
```

#### Exit Detection - Timeout Strategy
```
Frame 100: Employee NOT detected → Start absence timer
Frame 101-230: Employee NOT detected → Timer: 0.2s, 0.4s, ... 30s
Frame 231: Still absent → MARK EXIT ✓
```

#### Exit Detection - Frame-Based Strategy
```
Frame 100: Employee NOT detected → Absence count: 1
Frame 101-114: Employee NOT detected → Absence count: 2, 3, ... 15
Frame 115: Absence count = 15 → MARK EXIT ✓
```

#### Exit Detection - Hybrid Strategy (Recommended)
```
Frame 95-97: Employee detected near LEFT EDGE → Moving toward exit zone
Frame 98: Employee NOT detected (was in exit zone) → Fast exit threshold: 5 frames
Frame 99-102: Employee NOT detected → Absence count: 1, 2, 3, 4, 5
Frame 103: Absence count = 5 (fast threshold) → MARK EXIT ✓

OR (if person disappears from center):

Frame 100: Employee NOT detected (was in CENTER) → Normal exit threshold: 15 frames
Frame 101-114: Employee NOT detected → Absence count: 1, 2, ... 14, 15
Frame 115: Absence count = 15 → MARK EXIT ✓
```

## API Endpoints

### Edge Function: `mark-attendance`

**URL**: `{SUPABASE_URL}/functions/v1/mark-attendance`

**Method**: POST

**Headers**:
```
Authorization: Bearer {auth_token}
Content-Type: application/json
```

**Request Body**:
```json
{
  "employee_id": "uuid-of-employee",
  "action": "entry" | "exit"
}
```

**Response** (Success):
```json
{
  "success": true,
  "message": "Entry marked for John Doe",
  "action": "entry",
  "attendance_id": "uuid",
  "entry_time": "2024-01-07T10:30:00Z",
  "employee_name": "John Doe"
}
```

**Response** (Exit):
```json
{
  "success": true,
  "message": "Exit marked for John Doe",
  "action": "exit",
  "attendance_id": "uuid",
  "entry_time": "2024-01-07T10:30:00Z",
  "exit_time": "2024-01-07T17:45:00Z",
  "employee_name": "John Doe"
}
```

## Socket.IO Events

### `detection_frame` (Client → Server)

Sends video frame for detection. Response includes attendance actions:

```json
{
  "success": true,
  "detections": [...],
  "attendance_actions": [
    {
      "employee_id": "uuid",
      "employee_name": "John Doe",
      "action": "entry"
    }
  ]
}
```

### `stop_detection` (Client → Server)

Stops detection and resets attendance tracker state.

## Frontend Integration

The frontend (`LiveDetection.jsx`) receives attendance actions in the Socket.IO response:

```javascript
socket.on('detection_response', (data) => {
  if (data.attendance_actions && data.attendance_actions.length > 0) {
    data.attendance_actions.forEach(action => {
      console.log(`${action.employee_name} - ${action.action.toUpperCase()}`);
      // Show notification, update UI, etc.
    });
  }
});
```

## Adjusting Thresholds

### For Faster Entry Detection

If you want to mark entry faster (more sensitive):

```python
ATTENDANCE_CONFIG = {
    'MIN_CONSECUTIVE_DETECTIONS': 3,  # Reduced from 5
    'EXIT_STRATEGY': 'hybrid',
    'EXIT_FRAME_COUNT': 15,
    'EXIT_TIMEOUT_SECONDS': 30,
    'COOLDOWN_SECONDS': 5
}
```

### For More Reliable Entry Detection

If you want to avoid false entries:

```python
ATTENDANCE_CONFIG = {
    'MIN_CONSECUTIVE_DETECTIONS': 10,  # Increased from 5
    'EXIT_STRATEGY': 'hybrid',
    'EXIT_FRAME_COUNT': 15,
    'EXIT_TIMEOUT_SECONDS': 30,
    'COOLDOWN_SECONDS': 5
}
```

### For Faster Exit Detection

If employees leave quickly and you want faster exit marking:

```python
ATTENDANCE_CONFIG = {
    'MIN_CONSECUTIVE_DETECTIONS': 5,
    'EXIT_STRATEGY': 'hybrid',
    'EXIT_FRAME_COUNT': 10,  # Reduced from 15 (faster)
    'EXIT_TIMEOUT_SECONDS': 20,  # Reduced from 30
    'COOLDOWN_SECONDS': 5
}
```

### For More Reliable Exit Detection

If you want to avoid false exits (e.g., temporary occlusions):

```python
ATTENDANCE_CONFIG = {
    'MIN_CONSECUTIVE_DETECTIONS': 5,
    'EXIT_STRATEGY': 'hybrid',
    'EXIT_FRAME_COUNT': 20,  # Increased from 15 (more reliable)
    'EXIT_TIMEOUT_SECONDS': 60,  # Increased from 30
    'COOLDOWN_SECONDS': 5
}
```

### Switch to Simple Timeout Strategy

If you prefer the original simple approach:

```python
ATTENDANCE_CONFIG = {
    'MIN_CONSECUTIVE_DETECTIONS': 5,
    'EXIT_STRATEGY': 'timeout',  # Simple timeout
    'EXIT_TIMEOUT_SECONDS': 30,
    'COOLDOWN_SECONDS': 5
}
```

### Switch to Frame-Based Only

If you want consistency with entry logic:

```python
ATTENDANCE_CONFIG = {
    'MIN_CONSECUTIVE_DETECTIONS': 5,
    'EXIT_STRATEGY': 'frame_based',  # Frame counting only
    'EXIT_FRAME_COUNT': 15,
    'EXIT_TIMEOUT_SECONDS': 30,  # Not used but kept for fallback
    'COOLDOWN_SECONDS': 5
}
```

## Database Queries

### View Today's Attendance

```sql
SELECT 
    e.name,
    e.email,
    a.entry_time,
    a.exit_time,
    CASE 
        WHEN a.exit_time IS NULL THEN 'Present'
        ELSE 'Exited'
    END as status
FROM attendance a
JOIN employees e ON a.employee_id = e.id
WHERE DATE(a.entry_time) = CURRENT_DATE
ORDER BY a.entry_time DESC;
```

### View Employee Attendance History

```sql
SELECT 
    DATE(entry_time) as date,
    entry_time,
    exit_time,
    EXTRACT(EPOCH FROM (COALESCE(exit_time, NOW()) - entry_time))/3600 as hours_worked
FROM attendance
WHERE employee_id = 'uuid-of-employee'
ORDER BY entry_time DESC
LIMIT 30;
```

### Find Employees Currently Present

```sql
SELECT 
    e.name,
    e.email,
    a.entry_time,
    NOW() - a.entry_time as time_present
FROM attendance a
JOIN employees e ON a.employee_id = e.id
WHERE DATE(a.entry_time) = CURRENT_DATE
  AND a.exit_time IS NULL
ORDER BY a.entry_time;
```

## Troubleshooting

### Employees Not Being Marked as Present

1. **Check detection**: Verify employee is being recognized (green bounding box)
2. **Check threshold**: Increase `MIN_CONSECUTIVE_DETECTIONS` if too high, or check if employee is detected consistently
3. **Check logs**: Look for `[ATTENDANCE]` messages in backend console
4. **Check auth token**: Ensure valid Supabase auth token is being sent

### Duplicate Entries

1. **Check cooldown**: Increase `COOLDOWN_SECONDS` if entries are being marked too frequently
2. **Check database**: Verify edge function logic prevents duplicate entries for same day

### False Exits

1. **Increase timeout**: Increase `EXIT_TIMEOUT_SECONDS` if employees are being marked as exited too early
2. **Check detection**: Verify employee is consistently being detected (may be lighting/angle issues)

## Security

- Only `super_admin` and `hr_admin` roles can mark attendance
- All API calls require valid Supabase authentication token
- Edge function validates user role before processing requests
- Database constraints prevent invalid employee_id references

