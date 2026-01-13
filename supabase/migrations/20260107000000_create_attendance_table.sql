-- Create attendance table for tracking employee entry/exit
CREATE TABLE IF NOT EXISTS attendance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    employee_id UUID NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_attendance_employee_id ON attendance(employee_id);
CREATE INDEX IF NOT EXISTS idx_attendance_entry_time ON attendance(entry_time);
CREATE INDEX IF NOT EXISTS idx_attendance_exit_time ON attendance(exit_time) WHERE exit_time IS NOT NULL;

-- Create composite index for common queries (employee + entry_time)
-- Note: Date queries can use the entry_time index efficiently
CREATE INDEX IF NOT EXISTS idx_attendance_employee_entry ON attendance(employee_id, entry_time);

-- Add trigger to update updated_at timestamp
CREATE TRIGGER update_attendance_updated_at
    BEFORE UPDATE ON attendance
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE attendance IS 'Employee attendance records with entry and exit times';
COMMENT ON COLUMN attendance.employee_id IS 'Foreign key reference to employees table';
COMMENT ON COLUMN attendance.entry_time IS 'Timestamp when employee was marked as present';
COMMENT ON COLUMN attendance.exit_time IS 'Timestamp when employee was marked as exited (NULL if still present)';

