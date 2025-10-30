#!/usr/bin/env python3
"""
Initialize AlgoTradingbot Database
Creates all tables, indexes, and initial data
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'user': os.getenv('POSTGRES_USER', 'trading_user'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'database': os.getenv('POSTGRES_DB', 'trading'),
}

def get_connection(database='postgres'):
    """Get database connection"""
    config = DB_CONFIG.copy()
    config['database'] = database
    return psycopg2.connect(**config)

def database_exists():
    """Check if database exists"""
    try:
        conn = get_connection(database=DB_CONFIG['database'])
        conn.close()
        return True
    except psycopg2.OperationalError:
        return False

def create_database():
    """Create the trading database if it doesn't exist"""
    if database_exists():
        print(f"✓ Database '{DB_CONFIG['database']}' already exists")
        return

    print(f"Creating database '{DB_CONFIG['database']}'...")

    # Connect to postgres database to create new database
    conn = get_connection(database='postgres')
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    try:
        cursor.execute(
            sql.SQL("CREATE DATABASE {} WITH ENCODING 'UTF8' LC_COLLATE='C' LC_CTYPE='C' TEMPLATE=template0").format(
                sql.Identifier(DB_CONFIG['database'])
            )
        )
        print(f"✓ Database '{DB_CONFIG['database']}' created successfully")
    except psycopg2.errors.DuplicateDatabase:
        print(f"✓ Database '{DB_CONFIG['database']}' already exists")
    finally:
        cursor.close()
        conn.close()

def run_migration_file(filepath):
    """Run a SQL migration file"""
    print(f"Running migration: {filepath.name}")

    with open(filepath, 'r') as f:
        sql_content = f.read()

    conn = get_connection(database=DB_CONFIG['database'])
    cursor = conn.cursor()

    try:
        cursor.execute(sql_content)
        conn.commit()
        print(f"✓ Migration {filepath.name} completed successfully")
    except Exception as e:
        conn.rollback()
        print(f"✗ Error running migration {filepath.name}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def run_migrations():
    """Run all migration files in order"""
    migrations_dir = Path(__file__).parent.parent / 'sql' / 'migrations'

    if not migrations_dir.exists():
        print(f"✗ Migrations directory not found: {migrations_dir}")
        return

    # Get all .sql files and sort them
    migration_files = sorted(migrations_dir.glob('*.sql'))

    if not migration_files:
        print("✗ No migration files found")
        return

    print(f"\nFound {len(migration_files)} migration(s)")
    print("-" * 50)

    for migration_file in migration_files:
        run_migration_file(migration_file)

    print("-" * 50)

def verify_tables():
    """Verify that all expected tables exist"""
    expected_tables = [
        'market_data',
        'technical_indicators',
        'model_states',
        'predictions',
        'backtest_results',
        'backtest_trades',
        'system_logs',
        'configuration',
    ]

    print("\nVerifying tables...")

    conn = get_connection(database=DB_CONFIG['database'])
    cursor = conn.cursor()

    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)

    existing_tables = [row[0] for row in cursor.fetchall()]

    all_exist = True
    for table in expected_tables:
        if table in existing_tables:
            print(f"  ✓ {table}")
        else:
            print(f"  ✗ {table} - MISSING")
            all_exist = False

    cursor.close()
    conn.close()

    return all_exist

def get_table_stats():
    """Get statistics about database tables"""
    conn = get_connection(database=DB_CONFIG['database'])
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            schemaname as schema,
            tablename as table,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
    """)

    stats = cursor.fetchall()

    cursor.close()
    conn.close()

    return stats

def main():
    """Main initialization function"""
    print("=" * 50)
    print("AlgoTradingbot Database Initialization")
    print("=" * 50)

    # Check if password is set
    if not DB_CONFIG['password']:
        print("\n✗ ERROR: POSTGRES_PASSWORD environment variable is not set")
        print("Please set it in your .env file or environment")
        sys.exit(1)

    # Show configuration
    print(f"\nDatabase Configuration:")
    print(f"  Host: {DB_CONFIG['host']}")
    print(f"  Port: {DB_CONFIG['port']}")
    print(f"  User: {DB_CONFIG['user']}")
    print(f"  Database: {DB_CONFIG['database']}")

    try:
        # Step 1: Create database if needed
        print("\n" + "=" * 50)
        print("Step 1: Creating database")
        print("=" * 50)
        create_database()

        # Step 2: Run migrations
        print("\n" + "=" * 50)
        print("Step 2: Running migrations")
        print("=" * 50)
        run_migrations()

        # Step 3: Verify tables
        print("\n" + "=" * 50)
        print("Step 3: Verifying installation")
        print("=" * 50)
        all_tables_exist = verify_tables()

        # Step 4: Show statistics
        print("\n" + "=" * 50)
        print("Database Statistics")
        print("=" * 50)
        stats = get_table_stats()
        for schema, table, size in stats:
            print(f"  {table}: {size}")

        # Final message
        print("\n" + "=" * 50)
        if all_tables_exist:
            print("✓ Database initialization completed successfully!")
        else:
            print("⚠ Database initialization completed with warnings")
        print("=" * 50)

        return 0 if all_tables_exist else 1

    except Exception as e:
        print(f"\n✗ ERROR: Database initialization failed")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
