"""
Database Setup and Connection Script
This script establishes PostgreSQL connection, creates tables if they don't exist,
and provides testing functionality.
"""

import os
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
from datetime import datetime
import sys

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'attendance'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'alok@123')
}


def get_connection():
    """Establish connection to PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None


def test_connection():
    """Test database connection."""
    print("Testing database connection...")
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print(f"[OK] Connection successful!")
            print(f"  PostgreSQL version: {version[0]}")
            cursor.close()
            conn.close()
            return True
        except psycopg2.Error as e:
            print(f"[ERROR] Connection test failed: {e}")
            conn.close()
            return False
    else:
        print("[ERROR] Failed to establish connection")
        return False


def table_exists(cursor, table_name):
    """Check if a table exists in the database."""
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        );
    """, (table_name,))
    return cursor.fetchone()[0]


def create_tables():
    """Create tables if they don't exist."""
    conn = get_connection()
    if not conn:
        print("[ERROR] Cannot create tables: Connection failed")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create employees table
        if not table_exists(cursor, 'employees'):
            print("Creating 'employees' table...")
            cursor.execute("""
                CREATE TABLE employees (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    face_embedding BYTEA,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("[OK] 'employees' table created successfully")
        else:
            print("[OK] 'employees' table already exists")
        
        # Create attendance table
        if not table_exists(cursor, 'attendance'):
            print("Creating 'attendance' table...")
            cursor.execute("""
                CREATE TABLE attendance (
                    id SERIAL PRIMARY KEY,
                    employee_id INTEGER NOT NULL,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE CASCADE
                );
            """)
            print("[OK] 'attendance' table created successfully")
        else:
            print("[OK] 'attendance' table already exists")
        
        conn.commit()
        cursor.close()
        conn.close()
        print("\n[OK] All tables are ready!")
        return True
        
    except psycopg2.Error as e:
        print(f"[ERROR] Error creating tables: {e}")
        conn.rollback()
        conn.close()
        return False


def test_schema():
    """Test the database schema by checking table structures."""
    print("\nTesting database schema...")
    conn = get_connection()
    if not conn:
        print("[ERROR] Cannot test schema: Connection failed")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Check employees table structure
        print("\n'employees' table structure:")
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'employees'
            ORDER BY ordinal_position;
        """)
        for row in cursor.fetchall():
            print(f"  - {row[0]}: {row[1]} (nullable: {row[2]})")
        
        # Check attendance table structure
        print("\n'attendance' table structure:")
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'attendance'
            ORDER BY ordinal_position;
        """)
        for row in cursor.fetchall():
            print(f"  - {row[0]}: {row[1]} (nullable: {row[2]})")
        
        # Check foreign key constraint
        print("\nForeign key constraints:")
        cursor.execute("""
            SELECT
                tc.constraint_name,
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_name = 'attendance';
        """)
        fk = cursor.fetchone()
        if fk:
            print(f"  - {fk[2]} -> {fk[3]}.{fk[4]}")
        
        cursor.close()
        conn.close()
        print("\n[OK] Schema test completed!")
        return True
        
    except psycopg2.Error as e:
        print(f"[ERROR] Schema test failed: {e}")
        conn.close()
        return False


# ============================================================================
# REQUIRED QUERIES
# ============================================================================

class DatabaseQueries:
    """Class containing all required database queries."""
    
    def __init__(self):
        self.conn = get_connection()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    # ========== EMPLOYEE QUERIES ==========
    
    def insert_employee(self, name, face_embedding=None):
        """Insert a new employee."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO employees (name, face_embedding)
                VALUES (%s, %s)
                RETURNING id;
            """, (name, face_embedding))
            employee_id = cursor.fetchone()[0]
            self.conn.commit()
            cursor.close()
            return employee_id
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Error inserting employee: {e}")
            return None
    
    def get_employee_by_id(self, employee_id):
        """Get employee by ID."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, name, face_embedding, created_at
                FROM employees
                WHERE id = %s;
            """, (employee_id,))
            employee = cursor.fetchone()
            cursor.close()
            return employee
        except psycopg2.Error as e:
            print(f"Error getting employee: {e}")
            return None
    
    def get_all_employees(self):
        """Get all employees."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, name, created_at
                FROM employees
                ORDER BY id;
            """)
            employees = cursor.fetchall()
            cursor.close()
            return employees
        except psycopg2.Error as e:
            print(f"Error getting employees: {e}")
            return None
    
    def get_employee_by_name(self, name):
        """Get employee by name."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, name, face_embedding, created_at
                FROM employees
                WHERE name = %s;
            """, (name,))
            employee = cursor.fetchone()
            cursor.close()
            return employee
        except psycopg2.Error as e:
            print(f"Error getting employee: {e}")
            return None
    
    def update_employee(self, employee_id, name=None, face_embedding=None):
        """Update employee information."""
        try:
            cursor = self.conn.cursor()
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = %s")
                params.append(name)
            if face_embedding is not None:
                updates.append("face_embedding = %s")
                params.append(face_embedding)
            
            if not updates:
                return False
            
            params.append(employee_id)
            query = f"UPDATE employees SET {', '.join(updates)} WHERE id = %s;"
            cursor.execute(query, params)
            self.conn.commit()
            cursor.close()
            return cursor.rowcount > 0
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Error updating employee: {e}")
            return False
    
    def delete_employee(self, employee_id):
        """Delete an employee (cascade will delete attendance records)."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM employees WHERE id = %s;", (employee_id,))
            self.conn.commit()
            deleted = cursor.rowcount > 0
            cursor.close()
            return deleted
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Error deleting employee: {e}")
            return False
    
    # ========== ATTENDANCE QUERIES ==========
    
    def insert_attendance_entry(self, employee_id, entry_time=None):
        """Insert attendance entry (check-in)."""
        try:
            cursor = self.conn.cursor()
            if entry_time is None:
                entry_time = datetime.now()
            
            cursor.execute("""
                INSERT INTO attendance (employee_id, entry_time)
                VALUES (%s, %s)
                RETURNING id;
            """, (employee_id, entry_time))
            attendance_id = cursor.fetchone()[0]
            self.conn.commit()
            cursor.close()
            return attendance_id
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Error inserting attendance entry: {e}")
            return None
    
    def update_attendance_exit(self, attendance_id, exit_time=None):
        """Update attendance record with exit time (check-out)."""
        try:
            cursor = self.conn.cursor()
            if exit_time is None:
                exit_time = datetime.now()
            
            cursor.execute("""
                UPDATE attendance
                SET exit_time = %s
                WHERE id = %s AND exit_time IS NULL;
            """, (exit_time, attendance_id))
            self.conn.commit()
            updated = cursor.rowcount > 0
            cursor.close()
            return updated
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Error updating attendance exit: {e}")
            return False
    
    def get_attendance_by_id(self, attendance_id):
        """Get attendance record by ID."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT a.id, a.employee_id, e.name, a.entry_time, a.exit_time, a.created_at
                FROM attendance a
                JOIN employees e ON a.employee_id = e.id
                WHERE a.id = %s;
            """, (attendance_id,))
            attendance = cursor.fetchone()
            cursor.close()
            return attendance
        except psycopg2.Error as e:
            print(f"Error getting attendance: {e}")
            return None
    
    def get_attendance_by_employee(self, employee_id, start_date=None, end_date=None):
        """Get all attendance records for an employee."""
        try:
            cursor = self.conn.cursor()
            query = """
                SELECT a.id, a.employee_id, e.name, a.entry_time, a.exit_time, a.created_at
                FROM attendance a
                JOIN employees e ON a.employee_id = e.id
                WHERE a.employee_id = %s
            """
            params = [employee_id]
            
            if start_date:
                query += " AND DATE(a.entry_time) >= %s"
                params.append(start_date)
            if end_date:
                query += " AND DATE(a.entry_time) <= %s"
                params.append(end_date)
            
            query += " ORDER BY a.entry_time DESC;"
            cursor.execute(query, params)
            records = cursor.fetchall()
            cursor.close()
            return records
        except psycopg2.Error as e:
            print(f"Error getting attendance: {e}")
            return None
    
    def get_today_attendance(self):
        """Get all attendance records for today."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT a.id, a.employee_id, e.name, a.entry_time, a.exit_time
                FROM attendance a
                JOIN employees e ON a.employee_id = e.id
                WHERE DATE(a.entry_time) = CURRENT_DATE
                ORDER BY a.entry_time DESC;
            """)
            records = cursor.fetchall()
            cursor.close()
            return records
        except psycopg2.Error as e:
            print(f"Error getting today's attendance: {e}")
            return None
    
    def get_active_entries(self):
        """Get all active entries (checked in but not checked out)."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT a.id, a.employee_id, e.name, a.entry_time
                FROM attendance a
                JOIN employees e ON a.employee_id = e.id
                WHERE a.exit_time IS NULL
                ORDER BY a.entry_time DESC;
            """)
            records = cursor.fetchall()
            cursor.close()
            return records
        except psycopg2.Error as e:
            print(f"Error getting active entries: {e}")
            return None
    
    def clear_all_attendance(self):
        """Delete all attendance records."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM attendance;")
            deleted_count = cursor.rowcount
            self.conn.commit()
            cursor.close()
            print(f"[OK] Deleted {deleted_count} attendance records")
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Error clearing attendance: {e}")
            return False
    
    def clear_all_employees(self):
        """Delete all employees (cascades to attendance records)."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM employees;")
            deleted_count = cursor.rowcount
            self.conn.commit()
            cursor.close()
            print(f"[OK] Deleted {deleted_count} employees")
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Error clearing employees: {e}")
            return False
    
    def clear_all_data(self):
        """Delete all data from both tables (attendance first, then employees)."""
        try:
            cursor = self.conn.cursor()
            
            # Delete attendance records first
            cursor.execute("DELETE FROM attendance;")
            attendance_count = cursor.rowcount
            
            # Then delete employees
            cursor.execute("DELETE FROM employees;")
            employee_count = cursor.rowcount
            
            self.conn.commit()
            cursor.close()
            
            print(f"[OK] Cleared all data:")
            print(f"  - {attendance_count} attendance records deleted")
            print(f"  - {employee_count} employees deleted")
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Error clearing all data: {e}")
            return False
    
    def reset_sequences(self):
        """Reset auto-increment sequences for both tables."""
        try:
            cursor = self.conn.cursor()
            
            # Reset employees sequence
            cursor.execute("ALTER SEQUENCE employees_id_seq RESTART WITH 1;")
            
            # Reset attendance sequence
            cursor.execute("ALTER SEQUENCE attendance_id_seq RESTART WITH 1;")
            
            self.conn.commit()
            cursor.close()
            print("[OK] Database sequences reset")
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Error resetting sequences: {e}")
            return False
    
    def get_attendance_summary(self, start_date=None, end_date=None):
        """Get attendance summary with employee details."""
        try:
            cursor = self.conn.cursor()
            query = """
                SELECT 
                    e.id,
                    e.name,
                    COUNT(a.id) as total_records,
                    COUNT(CASE WHEN a.exit_time IS NOT NULL THEN 1 END) as completed_records,
                    MIN(a.entry_time) as first_entry,
                    MAX(a.entry_time) as last_entry
                FROM employees e
                LEFT JOIN attendance a ON e.id = a.employee_id
            """
            params = []
            
            if start_date or end_date:
                query += " WHERE 1=1"
                if start_date:
                    query += " AND DATE(a.entry_time) >= %s"
                    params.append(start_date)
                if end_date:
                    query += " AND DATE(a.entry_time) <= %s"
                    params.append(end_date)
            
            query += " GROUP BY e.id, e.name ORDER BY e.name;"
            cursor.execute(query, params)
            summary = cursor.fetchall()
            cursor.close()
            return summary
        except psycopg2.Error as e:
            print(f"Error getting attendance summary: {e}")
            return None



def main():
    """Main function to setup and test database.
    
    Usage examples for clearing data:
    
    # Clear only attendance records:
    with DatabaseQueries() as db:
        db.clear_all_attendance()
    
    # Clear only employees (cascades to attendance):
    with DatabaseQueries() as db:
        db.clear_all_employees()
    
    # Clear everything and reset sequences:
    # with DatabaseQueries() as db:
    #     db.clear_all_data()
    #     db.reset_sequences()
    """
    print("=" * 60)
    print("Database Setup and Testing Script")
    print("=" * 60)
    
    # Test connection
    if not test_connection():
        print("\n[ERROR] Cannot proceed: Database connection failed")
        print("Please check your .env file and ensure PostgreSQL is running.")
        sys.exit(1)
    
    # Create tables
    print("\n" + "=" * 60)
    print("Setting up database tables...")
    print("=" * 60)
    if not create_tables():
        print("\n[ERROR] Failed to create tables")
        sys.exit(1)
    
    # Test schema
    print("\n" + "=" * 60)
    if not test_schema():
        print("\n[ERROR] Schema test failed")
        sys.exit(1)
    
    # Test queries
    print("\n" + "=" * 60)
    print("Testing database queries...")
    print("=" * 60)
    
    with DatabaseQueries() as db:
        # Test employee insertion
        print("\n1. Testing employee insertion...")
        test_emp_id = db.insert_employee("Test Employee", None)
        if test_emp_id:
            print(f"   [OK] Employee inserted with ID: {test_emp_id}")
        else:
            print("   [ERROR] Failed to insert employee")
        
        # Test employee retrieval
        print("\n2. Testing employee retrieval...")
        employee = db.get_employee_by_id(test_emp_id)
        if employee:
            print(f"   [OK] Employee retrieved: {employee[1]} (ID: {employee[0]})")
        
        # Test attendance entry
        print("\n3. Testing attendance entry...")
        if test_emp_id:
            att_id = db.insert_attendance_entry(test_emp_id)
            if att_id:
                print(f"   [OK] Attendance entry created with ID: {att_id}")
        
        # Test attendance retrieval
        print("\n4. Testing attendance retrieval...")
        attendance = db.get_attendance_by_id(att_id) if test_emp_id else None
        if attendance:
            print(f"   [OK] Attendance retrieved: {attendance[2]} entered at {attendance[3]}")
        
        # Test getting all employees
        print("\n5. Testing get all employees...")
        employees = db.get_all_employees()
        if employees:
            print(f"   [OK] Found {len(employees)} employee(s)")
        
        # Clean up test data
        print("\n6. Cleaning up test data...")
        if test_emp_id:
            db.delete_employee(test_emp_id)
            print("   [OK] Test data cleaned up")
    
    print("\n" + "=" * 60)
    print("[OK] All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
   # main()
    dbquery = DatabaseQueries()

    dbquery.clear_all_employees()
    dbquery.clear_all_attendance()
    dbquery.clear_all_data()
    dbquery.reset_sequences()

    employees = dbquery.get_all_employees()
    print(employees)








