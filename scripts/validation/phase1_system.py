"""
Phase 1: System Setup Validation
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from .base import BaseValidator


class Phase1SystemValidator(BaseValidator):
    """Validator for system setup checks"""

    def __init__(self):
        super().__init__(1, "System Setup")
        self.project_root = Path(__file__).parent.parent.parent

    def validate(self):
        """Run all Phase 1 validations"""
        self.start()

        print(f"\n🔍 Phase 1: {self.phase.phase_name}")
        print("=" * 70)

        # Run all checks
        self._check_python_version()
        self._check_virtual_environment()
        self._check_required_packages()
        self._check_directory_structure()
        self._check_docker()
        self._check_docker_compose_services()
        self._check_environment_variables()

        self.end()
        return self.phase

    def _check_python_version(self):
        """Check Python version"""
        def check():
            version = sys.version_info
            required = (3, 11)
            meets_requirement = version >= required

            version_str = f"{version.major}.{version.minor}.{version.micro}"

            if meets_requirement:
                return True, f"Python {version_str}", {'version': version_str}
            else:
                return False, f"Python {version_str} (requires 3.11+)", {'version': version_str}

        result = self.run_check("Python Version", check)
        print(f"  {result.status.value} Python Version: {result.message}")

    def _check_virtual_environment(self):
        """Check if running in virtual environment"""
        def check():
            in_venv = hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            )

            if in_venv:
                return True, f"Active at {sys.prefix}", {'venv_path': sys.prefix}
            else:
                return False, "Not activated (recommended to use venv)", {}

        result = self.run_check("Virtual Environment", check)
        print(f"  {result.status.value} Virtual Environment: {result.message}")

    def _check_required_packages(self):
        """Check if required packages are installed"""
        def check():
            required_packages = [
                'pandas', 'numpy', 'tensorflow', 'xgboost', 'sklearn',
                'yfinance', 'psycopg2', 'redis', 'aiohttp', 'asyncio'
            ]

            missing = []
            for package in required_packages:
                try:
                    # Handle package name differences
                    import_name = package
                    if package == 'sklearn':
                        import_name = 'sklearn'
                    elif package == 'psycopg2':
                        import_name = 'psycopg2'

                    __import__(import_name)
                except ImportError:
                    missing.append(package)

            if not missing:
                return True, f"All {len(required_packages)} packages installed", {
                    'total': len(required_packages),
                    'missing': []
                }
            else:
                return False, f"Missing: {', '.join(missing)}", {
                    'total': len(required_packages),
                    'missing': missing
                }

        result = self.run_check("Required Packages", check)
        print(f"  {result.status.value} Required Packages: {result.message}")

    def _check_directory_structure(self):
        """Check if required directories exist"""
        def check():
            required_dirs = [
                'src', 'tests', 'scripts', 'sql/migrations',
                'data/raw', 'data/processed', 'models', 'logs', 'docs'
            ]

            missing = []
            for dir_path in required_dirs:
                full_path = self.project_root / dir_path
                if not full_path.exists():
                    missing.append(dir_path)

            if not missing:
                return True, f"All {len(required_dirs)} directories exist", {
                    'total': len(required_dirs),
                    'missing': []
                }
            else:
                return False, f"Missing: {', '.join(missing)}", {
                    'total': len(required_dirs),
                    'missing': missing
                }

        result = self.run_check("Directory Structure", check)
        print(f"  {result.status.value} Directory Structure: {result.message}")

    def _check_docker(self):
        """Check if Docker is running"""
        def check():
            try:
                result = subprocess.run(
                    ['docker', 'info'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    return True, "Docker daemon running", {}
                else:
                    return False, "Docker daemon not responding", {}

            except FileNotFoundError:
                return False, "Docker not installed", {}
            except subprocess.TimeoutExpired:
                return False, "Docker command timed out", {}
            except Exception as e:
                return False, str(e), {}

        result = self.run_check("Docker", check)
        print(f"  {result.status.value} Docker: {result.message}")

    def _check_docker_compose_services(self):
        """Check if docker-compose services are running"""
        def check():
            try:
                result = subprocess.run(
                    ['docker-compose', 'ps', '--services', '--filter', 'status=running'],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=self.project_root
                )

                if result.returncode == 0:
                    services = [s for s in result.stdout.strip().split('\n') if s]

                    if len(services) >= 2:
                        return True, f"{len(services)} services running", {
                            'services': services
                        }
                    else:
                        return False, f"Only {len(services)} service(s) running (expected 2+)", {
                            'services': services
                        }
                else:
                    return False, "Could not check services", {}

            except Exception as e:
                return False, str(e), {}

        result = self.run_check("Docker Compose Services", check)
        print(f"  {result.status.value} Docker Compose Services: {result.message}")

    def _check_environment_variables(self):
        """Check essential environment variables"""
        def check():
            # Load .env if exists
            from dotenv import load_dotenv
            load_dotenv(self.project_root / '.env')

            required_vars = ['POSTGRES_PASSWORD']
            missing = []

            for var in required_vars:
                if not os.getenv(var):
                    missing.append(var)

            if not missing:
                return True, "All required environment variables set", {
                    'required': required_vars,
                    'missing': []
                }
            else:
                return False, f"Missing: {', '.join(missing)}", {
                    'required': required_vars,
                    'missing': missing
                }

        result = self.run_check("Environment Variables", check)
        print(f"  {result.status.value} Environment Variables: {result.message}")
