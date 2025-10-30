#!/usr/bin/env python3
"""
Health Check Script for AlgoTradingbot
Verifies all system components are properly configured and operational
"""

import os
import sys
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Terminal colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.END}\n")

def print_check(name, status, message=""):
    """Print a check result"""
    if status:
        symbol = f"{Colors.GREEN}✓{Colors.END}"
        status_text = f"{Colors.GREEN}OK{Colors.END}"
    else:
        symbol = f"{Colors.RED}✗{Colors.END}"
        status_text = f"{Colors.RED}FAIL{Colors.END}"

    print(f"{symbol} {name:.<50} {status_text}")
    if message:
        print(f"  {Colors.YELLOW}↳ {message}{Colors.END}")

def check_python_version():
    """Check Python version"""
    try:
        version = sys.version_info
        required = (3, 11)
        meets_requirement = version >= required

        version_str = f"{version.major}.{version.minor}.{version.micro}"
        if meets_requirement:
            print_check("Python version", True, f"Python {version_str}")
        else:
            print_check("Python version", False, f"Python {version_str} (requires 3.11+)")

        return meets_requirement
    except Exception as e:
        print_check("Python version", False, str(e))
        return False

def check_virtual_environment():
    """Check if running in virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if in_venv:
        print_check("Virtual environment", True, f"Active at {sys.prefix}")
    else:
        print_check("Virtual environment", False, "Not activated (recommended to use venv)")

    return True  # Not critical

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        'pandas',
        'numpy',
        'tensorflow',
        'xgboost',
        'scikit-learn',
        'yfinance',
        'psycopg2',
        'redis',
        'aiohttp',
    ]

    all_installed = True
    missing = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
            all_installed = False

    if all_installed:
        print_check("Required packages", True, f"All {len(required_packages)} packages installed")
    else:
        print_check("Required packages", False, f"Missing: {', '.join(missing)}")

    return all_installed

def check_docker():
    """Check if Docker is running"""
    try:
        result = subprocess.run(
            ['docker', 'info'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print_check("Docker", True, "Docker daemon running")
            return True
        else:
            print_check("Docker", False, "Docker daemon not responding")
            return False

    except FileNotFoundError:
        print_check("Docker", False, "Docker not installed")
        return False
    except subprocess.TimeoutExpired:
        print_check("Docker", False, "Docker command timed out")
        return False
    except Exception as e:
        print_check("Docker", False, str(e))
        return False

def check_docker_compose():
    """Check if docker-compose services are running"""
    try:
        result = subprocess.run(
            ['docker-compose', 'ps', '--services', '--filter', 'status=running'],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).parent.parent
        )

        if result.returncode == 0:
            services = result.stdout.strip().split('\n')
            services = [s for s in services if s]  # Remove empty lines

            if len(services) >= 2:  # At least postgres and redis
                print_check("Docker Compose services", True, f"{len(services)} services running")
                return True
            elif len(services) > 0:
                print_check("Docker Compose services", False,
                          f"Only {len(services)} service(s) running (expected 2+)")
                return False
            else:
                print_check("Docker Compose services", False, "No services running")
                return False
        else:
            print_check("Docker Compose services", False, "Could not check services")
            return False

    except Exception as e:
        print_check("Docker Compose services", False, str(e))
        return False

def check_postgres():
    """Check PostgreSQL connection"""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            user=os.getenv('POSTGRES_USER', 'trading_user'),
            password=os.getenv('POSTGRES_PASSWORD'),
            database=os.getenv('POSTGRES_DB', 'trading'),
            connect_timeout=5
        )

        cursor = conn.cursor()
        cursor.execute('SELECT version();')
        version = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        print_check("PostgreSQL connection", True, "Connected successfully")
        return True

    except ImportError:
        print_check("PostgreSQL connection", False, "psycopg2 not installed")
        return False
    except Exception as e:
        print_check("PostgreSQL connection", False, str(e))
        return False

def check_postgres_tables():
    """Check if database tables exist"""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            user=os.getenv('POSTGRES_USER', 'trading_user'),
            password=os.getenv('POSTGRES_PASSWORD'),
            database=os.getenv('POSTGRES_DB', 'trading'),
            connect_timeout=5
        )

        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
        """)
        table_count = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        if table_count >= 8:  # Expecting 8 tables
            print_check("Database schema", True, f"{table_count} tables found")
            return True
        else:
            print_check("Database schema", False,
                      f"Only {table_count} tables (expected 8). Run init_database.py")
            return False

    except Exception as e:
        print_check("Database schema", False, str(e))
        return False

def check_redis():
    """Check Redis connection"""
    try:
        import redis

        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        r = redis.from_url(redis_url, socket_connect_timeout=5)

        # Test connection
        r.ping()

        # Test set/get
        r.set('health_check', 'ok', ex=10)
        value = r.get('health_check')

        r.close()

        if value == b'ok':
            print_check("Redis connection", True, "Connected and tested successfully")
            return True
        else:
            print_check("Redis connection", False, "Connection OK but test failed")
            return False

    except ImportError:
        print_check("Redis connection", False, "redis package not installed")
        return False
    except Exception as e:
        print_check("Redis connection", False, str(e))
        return False

def check_environment_variables():
    """Check essential environment variables"""
    required_vars = {
        'POSTGRES_PASSWORD': 'Database password',
    }

    optional_vars = {
        'ALPACA_API_KEY': 'Alpaca API key',
        'ALPACA_SECRET_KEY': 'Alpaca secret key',
    }

    all_set = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            print_check(f"Env var: {var}", True, description)
        else:
            print_check(f"Env var: {var}", False, f"{description} (REQUIRED)")
            all_set = False

    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print_check(f"Env var: {var}", True, description)
        else:
            print_check(f"Env var: {var}", False, f"{description} (optional)")

    return all_set

def check_directory_structure():
    """Check if required directories exist"""
    required_dirs = [
        'src',
        'tests',
        'scripts',
        'sql/migrations',
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'docs',
    ]

    project_root = Path(__file__).parent.parent
    all_exist = True

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print_check(f"Directory: {dir_path}", True)
        else:
            print_check(f"Directory: {dir_path}", False, "Missing")
            all_exist = False

    return all_exist

def check_tensorflow_gpu():
    """Check TensorFlow GPU/MPS support (Mac M4)"""
    try:
        import tensorflow as tf

        # Check for MPS (Metal Performance Shaders) on Mac
        devices = tf.config.list_physical_devices()
        has_gpu = any(d.device_type == 'GPU' for d in devices)

        if has_gpu:
            gpu_devices = [d for d in devices if d.device_type == 'GPU']
            print_check("TensorFlow GPU/MPS", True, f"{len(gpu_devices)} GPU device(s) available")
        else:
            print_check("TensorFlow GPU/MPS", False,
                      "No GPU/MPS devices (performance will be limited)")

        return True  # Not critical for basic functionality

    except Exception as e:
        print_check("TensorFlow GPU/MPS", False, str(e))
        return True  # Not critical

def main():
    """Main health check function"""
    print_header("AlgoTradingbot Health Check")

    checks = []

    # System checks
    print(f"\n{Colors.BOLD}System Environment{Colors.END}")
    checks.append(("python_version", check_python_version()))
    checks.append(("venv", check_virtual_environment()))
    checks.append(("packages", check_required_packages()))
    checks.append(("directories", check_directory_structure()))

    # Docker checks
    print(f"\n{Colors.BOLD}Docker Services{Colors.END}")
    checks.append(("docker", check_docker()))
    checks.append(("docker_compose", check_docker_compose()))

    # Database checks
    print(f"\n{Colors.BOLD}Database Services{Colors.END}")
    checks.append(("postgres", check_postgres()))
    checks.append(("postgres_tables", check_postgres_tables()))
    checks.append(("redis", check_redis()))

    # Configuration checks
    print(f"\n{Colors.BOLD}Configuration{Colors.END}")
    checks.append(("env_vars", check_environment_variables()))

    # Performance checks
    print(f"\n{Colors.BOLD}Performance{Colors.END}")
    checks.append(("tensorflow_gpu", check_tensorflow_gpu()))

    # Summary
    print_header("Health Check Summary")

    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    failed = total - passed

    print(f"Total Checks: {total}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.END}")
    if failed > 0:
        print(f"{Colors.RED}Failed: {failed}{Colors.END}")

    print()

    if failed == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All health checks passed!{Colors.END}")
        print(f"{Colors.GREEN}System is ready for development.{Colors.END}\n")
        return 0
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Some health checks failed{Colors.END}")
        print(f"{Colors.YELLOW}Please review the issues above and fix them.{Colors.END}\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
