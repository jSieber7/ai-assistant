#!/usr/bin/env python3

# =============================================================================
# Docker Services Runner Script
# =============================================================================
# This script starts Docker services with simplified command structure.
# Usage: uv run run_dockers.py [command] [options]
#   - Command: up (default), down, logs, status, reset, test, build
#   - Options: --service SERVICE (target specific service), --dev (dev mode, default is prod)
#              -f, --foreground (run in foreground mode, default is detached)
#              --no-cache (build without cache)
# =============================================================================
import os
import sys
import subprocess
import argparse
import re
import time
from pathlib import Path

# =============================================================================
# Configuration and Constants
# =============================================================================

# Get the directory where the script is located to ensure paths are correct
# regardless of where the script is called from.
SCRIPT_DIR = Path(__file__).parent.resolve()

# Color codes for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

# =============================================================================
# Helper Functions
# =============================================================================

def run_command(command, capture_output=False, check=True, foreground=False):
    """Runs a shell command, handling errors and output."""
    try:
        if foreground:
            # For interactive commands like `up -f` and `logs -f`
            process = subprocess.Popen(command, shell=True)
            process.wait()
            if process.returncode != 0:
                sys.exit(process.returncode)
        else:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=capture_output,
                text=True,
                check=check
            )
            return result
    except subprocess.CalledProcessError as e:
        # Error is already printed to stderr by default when check=True
        sys.exit(e.returncode)
    except FileNotFoundError:
        log_error(f"Command not found: {command.split()[0]}. Is it installed and in your PATH?")
        sys.exit(1)

def log_info(message):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def log_success(message):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

def log_warning(message):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def log_error(message):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}", file=sys.stderr)

def check_url_status(url):
    """Check if a URL returns a 404 status code."""
    if not re.match(r'^https?://', url):
        return ""
    
    command = f"curl -o /dev/null -s -w '%{{http_code}}' --max-time 5 '{url}' 2>/dev/null || echo '000'"
    result = run_command(command, capture_output=True, check=False)
    status_code = result.stdout.strip()
    
    # Clean up the status code to ensure it's just the numeric value
    status_code = re.sub(r'[^0-9]', '', status_code)
    
    if status_code == "404":
        return " (404)"
    elif status_code == "200":
        return " (OK)"
    elif status_code == "000" or status_code == "":
        return " (Error)"
    else:
        return f" ({status_code})"

def get_all_services():
    """Get a list of all available services by finding docker-compose files."""
    services = set()  # Use a set to avoid duplicates
    for item in SCRIPT_DIR.iterdir():
        if item.is_dir():
            # Check for default compose files
            if (item / "docker-compose.yml").exists() or (item / "docker-compose.yaml").exists():
                services.add(item.name)
            # Check for environment-specific compose files
            elif (item / "docker-compose.dev.yml").exists() or (item / "docker-compose.dev.yaml").exists():
                services.add(item.name)
            elif (item / "docker-compose.prod.yml").exists() or (item / "docker-compose.prod.yaml").exists():
                services.add(item.name)
    
    # Check for frontend in docker directory (special case)
    frontend_dir = SCRIPT_DIR / "frontend"
    if frontend_dir.is_dir():
        # Check for environment-specific compose files
        if (frontend_dir / "docker-compose.dev.yml").exists() or (frontend_dir / "docker-compose.dev.yaml").exists():
            services.add("frontend")
        elif (frontend_dir / "docker-compose.prod.yml").exists() or (frontend_dir / "docker-compose.prod.yaml").exists():
            services.add("frontend")
        # Check for default compose files
        elif (frontend_dir / "docker-compose.yml").exists() or (frontend_dir / "docker-compose.yaml").exists():
            services.add("frontend")
    
    return sorted(list(services))

def find_compose_file(service_dir, environment=None):
    """Find the docker-compose file in a given service directory.
    
    For services with environment-specific compose files, it will look for:
    - docker-compose.{environment}.yml/yaml
    - docker-compose.yml/yaml (fallback)
    
    Args:
        service_dir: The service directory name
        environment: The environment (dev, prod) to look for environment-specific compose files
        
    Returns:
        The compose file name or None if not found
    """
    # Special case for frontend service
    if service_dir == "frontend":
        frontend_dir = SCRIPT_DIR / "frontend"
        # First try environment-specific compose files
        if environment:
            env_yml_path = frontend_dir / f"docker-compose.{environment}.yml"
            env_yaml_path = frontend_dir / f"docker-compose.{environment}.yaml"
            if env_yml_path.exists():
                return env_yml_path.name
            if env_yaml_path.exists():
                return env_yaml_path.name
        
        # Fallback to default compose files
        yml_path = frontend_dir / "docker-compose.yml"
        yaml_path = frontend_dir / "docker-compose.yaml"
        if yml_path.exists():
            return yml_path.name
        if yaml_path.exists():
            return yaml_path.name
        return None
    
    # First try environment-specific compose files
    if environment:
        env_yml_path = SCRIPT_DIR / service_dir / f"docker-compose.{environment}.yml"
        env_yaml_path = SCRIPT_DIR / service_dir / f"docker-compose.{environment}.yaml"
        if env_yml_path.exists():
            return env_yml_path.name
        if env_yaml_path.exists():
            return env_yaml_path.name
    
    # Fallback to default compose files
    yml_path = SCRIPT_DIR / service_dir / "docker-compose.yml"
    yaml_path = SCRIPT_DIR / service_dir / "docker-compose.yaml"
    if yml_path.exists():
        return yml_path.name
    if yaml_path.exists():
        return yaml_path.name
    return None

def check_docker():
    """Check if Docker is running."""
    if run_command("docker info > /dev/null 2>&1", check=False).returncode != 0:
        log_error("Docker is not running. Please start Docker and try again.")
        sys.exit(1)

def validate_inputs(service, environment):
    """Validate the provided service and environment."""
    services = get_all_services()
    
    if service == "all":
        if environment not in ["dev", "prod"]:
            log_error(f"Invalid environment '{environment}'. Use 'dev' or 'prod'.")
            sys.exit(1)
        
        for s in services:
            # Special case for frontend service
            if s == "frontend":
                env_file = SCRIPT_DIR / "frontend" / "env" / f"{environment}.env"
            else:
                env_file = SCRIPT_DIR / s / "env" / f"{environment}.env"
            
            if not env_file.is_file():
                log_error(f"Environment file '{env_file}' does not exist.")
                sys.exit(1)
        
        log_info("Using all services")
        log_info(f"Using environment: {environment}")
        return

    if service not in services:
        log_error(f"Service directory '{service}' does not exist.")
        log_info("Available services:")
        for s in services:
            log_info(f"  - {s}")
        log_info("  - all (run all services)")
        sys.exit(1)

    compose_file = find_compose_file(service, environment)
    if not compose_file:
        log_error(f"Service '{service}' does not contain a docker-compose file for environment '{environment}'.")
        sys.exit(1)

    if environment not in ["dev", "prod"]:
        log_error(f"Invalid environment '{environment}'. Use 'dev' or 'prod'.")
        sys.exit(1)

    # Special case for frontend service
    if service == "frontend":
        env_file = SCRIPT_DIR / "frontend" / "env" / f"{environment}.env"
    else:
        env_file = SCRIPT_DIR / service / "env" / f"{environment}.env"
    
    if not env_file.is_file():
        log_error(f"Environment file '{env_file}' does not exist.")
        sys.exit(1)

    log_info(f"Using service: {service}")
    log_info(f"Using environment: {environment}")
    log_info(f"Using compose file: {service}/{compose_file}")
    return compose_file

def is_traefik_running():
    """Check if traefik service is running."""
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    result = run_command("docker compose -f docker-compose.yml ps traefik --format 'table {{.Status}}'", capture_output=True, check=False)
    os.chdir(original_cwd)
    return result.returncode == 0 and "running" in result.stdout.lower()

def get_service_info(service, environment="dev"):
    """Get service-specific information like URLs."""
    # Check if traefik is running to determine which URLs to show
    traefik_running = is_traefik_running()
    
    if traefik_running:
        # Use traefik URLs when traefik is running
        service_urls = {
            "supabase": [
                "Studio & API: http://supabase.localhost (Kong Gateway routes to Studio and API)",
                "Database: postgresql://postgres:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT:-5432}/postgres"
            ],
            "firecrawl": [
                "API: http://firecrawl.localhost",
                "Redis: redis://localhost:6379",
                "Database: postgresql://postgres:postgres@localhost:5432/postgres"
            ],
            "searxng": [
                "Search Engine: http://searxng.localhost"
            ],
            "frontend": [
                f"React App: http://frontend.localhost (routes to {'dev' if environment == 'dev' else 'prod'} service on port {3000 if environment == 'dev' else 80})"
            ],
            "app": [
                "FastAPI App: http://app.localhost"
            ],
            "traefik": [
                "Dashboard: http://traefik.localhost"
            ]
        }
    else:
        # Use direct port URLs when traefik is not running
        service_urls = {
            "supabase": [
                "Studio & API: Not directly accessible (requires Traefik: supabase.localhost)",
                "Database: postgresql://postgres:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT:-5432}/postgres"
            ],
            "firecrawl": [
                "API: Not directly accessible (requires Traefik: firecrawl.localhost)",
                "Redis: redis://localhost:6379",
                "Database: postgresql://postgres:postgres@localhost:5432/postgres"
            ],
            "searxng": [
                "Search Engine: Not directly accessible (requires Traefik: searxng.localhost)"
            ],
            "frontend": [
                f"React App: http://localhost:{3000 if environment == 'dev' else 80} ({'dev mode' if environment == 'dev' else 'prod mode'})"
            ],
            "app": [
                "FastAPI App: http://localhost:8000"
            ],
            "traefik": [
                "Dashboard: http://localhost:8881"
            ]
        }
    return service_urls.get(service, ["Service running on default ports"])

def display_service_urls(service, environment="dev"):
    """Display the URLs for a given service."""
    # Wait a moment for services to fully start
    log_info("Waiting for services to fully start...")
    time.sleep(3)
    
    urls = get_service_info(service, environment)
    for url_info in urls:
        url_part_match = re.search(r'https?://[^\s]+', url_info)
        status_indicator = ""
        if url_part_match:
            status_indicator = check_url_status(url_part_match.group(0))
        log_info(f"{url_info}{status_indicator}")

# =============================================================================
# Single Service Command Functions
# =============================================================================

def start_services(service, environment, compose_file, foreground):
    log_info(f"Starting {service} services in {environment} mode...")
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    
    # First, stop any running services to ensure clean start
    log_info(f"Stopping any running {service} services...")
    run_command(f"docker compose --env-file {service}/env/{environment}.env -f docker-compose.yml stop {service}", check=False)
    
    # Use the main docker-compose.yml in the docker folder
    compose_cmd = f"docker compose --env-file {service}/env/{environment}.env -f docker-compose.yml"
    if foreground:
        run_command(f"{compose_cmd} up {service}", foreground=True)
    else:
        run_command(f"{compose_cmd} up -d {service}")
        log_success(f"{service} services started in detached mode!")
        display_service_urls(service, environment)
        log_info(f"Use 'uv run run_dockers.py -s {service} logs' to view logs")
    os.chdir(original_cwd)

def stop_services(service, environment, compose_file):
    log_info(f"Stopping {service} services...")
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    run_command(f"docker compose --env-file {service}/env/{environment}.env -f docker-compose.yml stop {service}")
    log_success(f"{service} services stopped.")
    os.chdir(original_cwd)

def show_logs(service, environment, compose_file):
    log_info(f"Showing {service} logs...")
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    run_command(f"docker compose --env-file {service}/env/{environment}.env -f docker-compose.yml logs -f {service}", foreground=True)
    os.chdir(original_cwd)

def show_status(service, environment, compose_file):
    log_info(f"Checking {service} service status...")
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    run_command(f"docker compose --env-file {service}/env/{environment}.env -f docker-compose.yml ps {service}")
    os.chdir(original_cwd)

def reset_services(service, environment, compose_file):
    log_warning(f"This will reset all {service} data. Are you sure? (y/N)")
    confirmation = input().strip().lower()
    if confirmation not in ['y', 'yes']:
        log_info("Reset cancelled.")
        sys.exit(0)
    
    log_info(f"Resetting {service}...")
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    run_command(f"docker compose --env-file {service}/env/{environment}.env -f docker-compose.yml rm -f -v {service}")
    log_success("Reset complete.")
    os.chdir(original_cwd)

def test_configuration(service, environment, compose_file):
    log_info(f"Running configuration tests for {service} in {environment} environment...")
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    if run_command(f"docker compose --env-file {service}/env/{environment}.env -f docker-compose.yml config {service} > /dev/null", check=False).returncode == 0:
        log_success("Configuration is valid!")
    else:
        log_error("Configuration validation failed.")
        os.chdir(original_cwd)
        sys.exit(1)
    os.chdir(original_cwd)

def build_service(service, environment, compose_file, no_cache=False, progress="auto"):
    log_info(f"Building {service} services in {environment} mode...")
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    
    # Use the main docker-compose.yml in the docker folder
    compose_cmd = f"docker compose --env-file {service}/env/{environment}.env -f docker-compose.yml build"
    if no_cache:
        compose_cmd += " --no-cache"
    if progress != "auto":
        compose_cmd += f" --progress={progress}"
    compose_cmd += f" {service}"
    
    run_command(compose_cmd)
    log_success(f"{service} services built successfully!")
    os.chdir(original_cwd)

# =============================================================================
# All Services Command Functions
# =============================================================================

def start_all_services(environment, foreground):
    services = get_all_services()
    log_info(f"Starting all services in {environment} mode...")
    
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    
    # First, stop any running services to ensure clean start
    log_info("Stopping any running services...")
    compose_stop_cmd = f"docker compose"
    for service in services:
        env_file = SCRIPT_DIR / service / "env" / f"{environment}.env"
        if env_file.is_file():
            compose_stop_cmd += f" --env-file {service}/env/{environment}.env"
    
    compose_stop_cmd += " -f docker-compose.yml down"
    run_command(compose_stop_cmd, check=False)
    
    # Start all services using the main docker-compose.yml
    compose_cmd = f"docker compose"
    for service in services:
        env_file = SCRIPT_DIR / service / "env" / f"{environment}.env"
        if env_file.is_file():
            compose_cmd += f" --env-file {service}/env/{environment}.env"
    
    compose_cmd += " -f docker-compose.yml"
    
    # Load main .env file if it exists
    main_env_file = Path(__file__).parent.parent / ".env"
    if main_env_file.is_file():
        compose_cmd += f" --env-file {main_env_file}"
    
    if foreground:
        run_command(f"{compose_cmd} up", foreground=True)
    else:
        run_command(f"{compose_cmd} up -d")
        log_success("All services started in detached mode!")
        display_all_service_urls(environment)
        log_info(f"Use 'uv run run_dockers.py logs' to view logs")
    
    os.chdir(original_cwd)

def stop_all_services(environment):
    services = get_all_services()
    log_info("Stopping all services...")
    
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    
    # Stop all services using the main docker-compose.yml
    compose_cmd = f"docker compose"
    for service in services:
        env_file = SCRIPT_DIR / service / "env" / f"{environment}.env"
        if env_file.is_file():
            compose_cmd += f" --env-file {service}/env/{environment}.env"
    
    compose_cmd += " -f docker-compose.yml down"
    run_command(compose_cmd)
    
    os.chdir(original_cwd)
    log_success("All services stopped.")

def show_all_logs(environment, foreground):
    if foreground:
        log_error("Cannot follow logs for all services at once. Please specify a single service.")
        log_info("Example: uv run run_dockers.py -s supabase logs")
        sys.exit(1)

    services = get_all_services()
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    
    # Show logs for all services using the main docker-compose.yml
    compose_cmd = f"docker compose"
    for service in services:
        env_file = SCRIPT_DIR / service / "env" / f"{environment}.env"
        if env_file.is_file():
            compose_cmd += f" --env-file {service}/env/{environment}.env"
    
    compose_cmd += " -f docker-compose.yml logs --tail=50"
    run_command(compose_cmd)
    
    os.chdir(original_cwd)

def show_all_status(environment):
    services = get_all_services()
    log_info("Checking status for all services...")
    
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    
    # Show status for all services using the main docker-compose.yml
    compose_cmd = f"docker compose"
    for service in services:
        env_file = SCRIPT_DIR / service / "env" / f"{environment}.env"
        if env_file.is_file():
            compose_cmd += f" --env-file {service}/env/{environment}.env"
    
    compose_cmd += " -f docker-compose.yml ps"
    run_command(compose_cmd)
    
    os.chdir(original_cwd)

def reset_all_services(environment):
    services = get_all_services()
    log_warning("This will reset all service data. Are you sure? (y/N)")
    confirmation = input().strip().lower()
    if confirmation not in ['y', 'yes']:
        log_info("Reset cancelled.")
        sys.exit(0)

    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    
    # Reset all services using the main docker-compose.yml
    compose_cmd = f"docker compose"
    for service in services:
        env_file = SCRIPT_DIR / service / "env" / f"{environment}.env"
        if env_file.is_file():
            compose_cmd += f" --env-file {service}/env/{environment}.env"
    
    compose_cmd += " -f docker-compose.yml down -v --remove-orphans"
    run_command(compose_cmd)
    
    os.chdir(original_cwd)
    log_success("All services reset complete.")

def test_all_configuration(environment):
    services = get_all_services()
    log_info(f"Running configuration tests for all services in {environment} environment...")
    
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    
    # Test configuration for all services using the main docker-compose.yml
    compose_cmd = f"docker compose"
    for service in services:
        env_file = SCRIPT_DIR / service / "env" / f"{environment}.env"
        if env_file.is_file():
            compose_cmd += f" --env-file {service}/env/{environment}.env"
    
    compose_cmd += " -f docker-compose.yml config > /dev/null"
    if run_command(compose_cmd, check=False).returncode == 0:
        log_success("All service configurations are valid!")
    else:
        log_error("Configuration validation failed.")
        os.chdir(original_cwd)
        sys.exit(1)
    
    os.chdir(original_cwd)

def build_all_services(environment, no_cache=False, progress="auto"):
    services = get_all_services()
    log_info(f"Building all services in {environment} mode...")
    
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    
    # Build all services using the main docker-compose.yml
    compose_cmd = f"docker compose"
    for service in services:
        env_file = SCRIPT_DIR / service / "env" / f"{environment}.env"
        if env_file.is_file():
            compose_cmd += f" --env-file {service}/env/{environment}.env"
    
    compose_cmd += " -f docker-compose.yml build"
    if no_cache:
        compose_cmd += " --no-cache"
    if progress != "auto":
        compose_cmd += f" --progress={progress}"
    
    run_command(compose_cmd)
    log_success("All services built successfully!")
    os.chdir(original_cwd)

def display_all_service_urls(environment):
    services = get_all_services()
    # Always display Traefik URL since it's defined in the main compose file
    log_info(f"URLs for traefik service:")
    display_service_urls("traefik", environment)
    print()
    
    for service in services:
        log_info(f"URLs for {service} service:")
        display_service_urls(service, environment)
        print()

# =============================================================================
# Main Execution
# =============================================================================

def show_help():
    """Show help information."""
    print(f"{Colors.BLUE}Docker Services Management Script{Colors.NC}")
    print("")
    print(f"{Colors.YELLOW}USAGE:{Colors.NC}")
    print("  uv run run_dockers.py [command] [options]")
    print("  uv run run_dockers.py --help | -h")
    print("")
    print(f"{Colors.YELLOW}COMMANDS:{Colors.NC}")
    print("  up           Start services (default)")
    print("  down         Stop services")
    print("  logs         Show service logs")
    print("  status       Show service status")
    print("  reset        Reset service data")
    print("  test         Test service configuration")
    print("  build        Build services")
    print("")
    print(f"{Colors.YELLOW}OPTIONS:{Colors.NC}")
    print("  --service, -s SERVICE    Target specific service (default: all)")
    print("  --dev, -d                 Use development environment (default: production)")
    print("  -f, --foreground    Run in foreground mode (default is detached)")
    print("  --no-cache          Build without using cache (for build command only)")
    print("  --progress MODE      Set build progress output mode (auto, plain, tty)")
    print("  --help, -h          Show this help message")
    print("")
    print(f"{Colors.YELLOW}EXAMPLES:{Colors.NC}")
    print("  uv run run_dockers.py                    # Start all services in prod mode (detached)")
    print("  uv run run_dockers.py -s firecrawl -d up  # Start firecrawl in dev mode (detached)")
    print("  uv run run_dockers.py -s supabase down    # Stop supabase in prod mode")
    print("  uv run run_dockers.py -s firecrawl -f up   # Start firecrawl in foreground mode")
    print("  uv run run_dockers.py -s firecrawl logs    # Show firecrawl logs")
    print("  uv run run_dockers.py -d down              # Stop all services in dev mode")
    print("  uv run run_dockers.py -s firecrawl build   # Build firecrawl service")
    print("  uv run run_dockers.py build --no-cache              # Build all services without cache")
    print("")
    print(f"{Colors.YELLOW}AVAILABLE SERVICES:{Colors.NC}")
    services = get_all_services()
    for service in services:
        print(f"  - {service}")

def main():
    """Main function to parse arguments and execute commands."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Docker Services Management Script",
        add_help=False  # We'll handle help manually to show our custom help
    )
    
    # Command argument (positional, with default)
    parser.add_argument('command', nargs='?', default='up', 
                       choices=['up', 'down', 'logs', 'status', 'reset', 'test', 'build'],
                       help="Command to execute (default: up)")
    
    # Optional flags
    parser.add_argument('--service', '-s', type=str, default='all',
                       help="Target specific service (default: all)")
    parser.add_argument('--dev', '-d', action='store_true',
                       help="Use development environment (default: production)")
    parser.add_argument('-f', '--foreground', action='store_true',
                       help="Run in foreground mode (default is detached)")
    parser.add_argument('--no-cache', action='store_true',
                       help="Build without using cache (for build command only)")
    parser.add_argument('--progress', type=str, default='auto', choices=['auto', 'plain', 'tty'],
                       help="Set build progress output mode (default: auto)")
    parser.add_argument('--help', '-h', action='store_true',
                       help="Show this help message")
    
    args = parser.parse_args()
    
    # Show help if requested
    if args.help:
        show_help()
        sys.exit(0)
    
    # Determine environment based on --dev flag
    environment = "dev" if args.dev else "prod"
    
    # Run checks and setup
    check_docker()
    compose_file = validate_inputs(args.service, environment)

    # Execute command
    if args.service == "all":
        if args.command == "up":
            start_all_services(environment, args.foreground)
        elif args.command == "down":
            stop_all_services(environment)
        elif args.command == "logs":
            show_all_logs(environment, args.foreground)
        elif args.command == "status":
            show_all_status(environment)
        elif args.command == "reset":
            reset_all_services(environment)
        elif args.command == "test":
            test_all_configuration(environment)
        elif args.command == "build":
            build_all_services(environment, args.no_cache, args.progress)
    else:
        if args.command == "up":
            start_services(args.service, environment, compose_file, args.foreground)
        elif args.command == "down":
            stop_services(args.service, environment, compose_file)
        elif args.command == "logs":
            show_logs(args.service, environment, compose_file)
        elif args.command == "status":
            show_status(args.service, environment, compose_file)
        elif args.command == "reset":
            reset_services(args.service, environment, compose_file)
        elif args.command == "test":
            test_configuration(args.service, environment, compose_file)
        elif args.command == "build":
            build_service(args.service, environment, compose_file, args.no_cache, args.progress)

if __name__ == "__main__":
    main()