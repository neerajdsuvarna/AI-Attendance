# Face Recognition Attendance System

## Quick Start

```bash
conda activate opencv-env13
python attendance_system.py  # Main production system
```

## Files Overview

### 📁 Core Files
- **`attendance_system.py`** ⭐ - **USE THIS!** Clean, optimized, single-file system
- `entry_exit.py` - Original multi-face version (verbose)
- `RegisterFace.py` - Register new employees
- `setup_database.py` - Database setup and utilities

### 🗄️ Database
- PostgreSQL database (config in `.env`)
- Tables: `employees`, `attendance`

## Features

✅ **Multi-face detection** - Track multiple people simultaneously
✅ **Entry/Exit tracking** - Automatic detection with 5-min cooldown
✅ **Real-time recognition** - Fast processing (every 5th frame)
✅ **Visual feedback** - Color-coded bounding boxes
✅ **PostgreSQL storage** - Persistent attendance records

## Usage

### 1. Register New Employee
```bash
python RegisterFace.py
# Press SPACE to capture face, enter name
```

### 2. Run Attendance System
```bash
python attendance_system.py
# System auto-detects and tracks faces
# Press ESC to exit
```

## Configuration

Edit top of `attendance_system.py`:

```python
SIMILARITY_THRESHOLD = 0.4  # Recognition accuracy (0.3-0.5)
MIN_TIME_GAP = timedelta(minutes=5)  # Entry/exit cooldown
PROCESS_EVERY_N_FRAMES = 5  # Speed (lower = faster but more CPU)
```

## Color Codes

- 🟢 **Green** - Entry marked or recognized
- 🔴 **Red** - Exit marked or unknown person
- 🟡 **Yellow** - Already logged (cooldown)

## Troubleshooting

**Camera not working?**
- Close apps using camera (Zoom, Teams, Skype)
- Check Windows camera permissions
- System auto-tries indices 0, 1, 2

**Low recognition accuracy?**
- Lower `SIMILARITY_THRESHOLD` to 0.3
- Ensure good lighting
- Register face from multiple angles

**Slow performance?**
- Increase `PROCESS_EVERY_N_FRAMES` to 10
- Close other apps
- Use GPU (change to `CUDAExecutionProvider`)

## Database Queries

Check attendance records:
```sql
-- Today's attendance
SELECT e.name, a.entry_time, a.exit_time 
FROM attendance a 
JOIN employees e ON a.employee_id = e.id 
WHERE DATE(a.entry_time) = CURRENT_DATE;

-- All employees
SELECT * FROM employees;
```

## Architecture

**Why single-file?**
- ✅ Easier to debug
- ✅ Faster to modify
- ✅ No import issues
- ✅ Clear code flow
- ✅ Production-ready

**Core Components:**
1. **Face Detection** - SCRFD model (640x640)
2. **Face Recognition** - ArcFace/GLintr100 (512-d embeddings)
3. **Database** - PostgreSQL with psycopg2
4. **Logic** - Entry/exit state machine

## Performance

- **FPS**: ~20-30 (processing every 5th frame)
- **Detection**: ~50-100ms per frame
- **Recognition**: ~20-30ms per face
- **Max faces**: 10 simultaneous

## Models Used

Located in: `C:\Users\Alok\.insightface\models\antelopev2\`
- `scrfd_10g_bnkps.onnx` - Face detection
- `glintr100.onnx` - Face recognition (512-d embeddings)

## License & Credits

- InsightFace models (Apache 2.0)
- OpenCV (Apache 2.0)
- ONNX Runtime (MIT)

