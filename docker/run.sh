#!/bin/bash

# =============================================================================
# Hybrid Docker Compose Setup - Complete Runner Script
# =============================================================================
# This script starts all services, performs health checks, and provides
# comprehensive status reporting for the hybrid Docker Compose setup.
# Usage: ./run.sh [dev|prod] [command]
#   - Environment: dev (default) or prod
#   - Command: up (default), down, logs, etc.
# =============================================================================

set -e  # Exit on any error

# Default to 'dev' environment and 'up' command
ENVIRONMENT=${1:-dev}
COMMAND=${2:-up}

# Validate environment
if [[ "$ENVIRONMENT" != "dev" && "$ENVIRONMENT" != "prod" ]]; then
    echo "Error: Invalid environment '$ENVIRONMENT'. Use 'dev' or 'prod'."
    exit 1
fi

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service configuration
SERVICES=("traefik" "redis" "fast-api-app" "firecrawl-postgres" "firecrawl-playwright" "firecrawl-api" "searxng" "supabase-db" "supabase-kong" "supabase-auth" "supabase-rest" "supabase-realtime" "supabase-storage" "supabase-studio" "supabase-analytics" "supabase-vector" "supabase-pooler")
SERVICE_URLS=(
    ["traefik"]="http://traefik.localhost:8080"
    ["fast-api-app"]="http://fastapi.localhost"
    ["firecrawl"]="http://firecrawl.localhost"
    ["searxng"]="http://searxng.localhost"
)
HEALTH_CHECK_TIMEOUT=60
HEALTH_CHECK_INTERVAL=5

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Environment Setup
# =============================================================================

setup_environment() {
    log_info "Setting up '$ENVIRONMENT' environment..."
    
    # Check if base .env file exists
    if [ ! -f ".env" ]; then
        log_error "Main .env file not found!"
        exit 1
    fi

    # Check if service-specific .env files exist
    local services=("firecrawl" "searxng" "supabase")
    for service in "${services[@]}"; do
        if [ ! -f "env/${service}.env" ]; then
            log_warning "Service .env file not found: env/${service}.env. Please create it."
        fi
        if [ ! -f "env/${service}.${ENVIRONMENT}.env" ]; then
            log_warning "Environment override file not found: env/${service}.${ENVIRONMENT}.env. Please create it."
        fi
    done
    
    log_success "Environment setup complete"
}

# =============================================================================
# Service Management
# =============================================================================

get_docker_compose_cmd() {
    local compose_files="-f docker-compose.yml"
    if [ -f "docker-compose.${ENVIRONMENT}.yml" ]; then
        compose_files="$compose_files -f docker-compose.${ENVIRONMENT}.yml"
    fi

    local env_files="--env-file .env"
    local services=("firecrawl" "searxng" "supabase")
    for service in "${services[@]}"; do
        if [ -f "env/${service}.env" ]; then
            env_files="$env_files --env-file env/${service}.env"
        fi
        if [ -f "env/${service}.${ENVIRONMENT}.env" ]; then
            env_files="$env_files --env-file env/${service}.${ENVIRONMENT}.env"
        fi
    done
    
    echo "docker compose $compose_files $env_files"
}

start_services() {
    log_info "Starting all services for '$ENVIRONMENT' environment..."
    
    local compose_cmd=$(get_docker_compose_cmd)
    
    # Stop any existing services
    log_info "Stopping any existing services..."
    $compose_cmd down --remove-orphans 2>/dev/null || true
    
    # Start services
    log_info "Starting services with Docker Compose..."
    if $compose_cmd up -d --quiet-pull; then
        log_success "All services started successfully"
    else
        log_error "Failed to start services"
        exit 1
    fi
}

# =============================================================================
# Health Checks
# =============================================================================

check_service_health() {
    local service=$1
    local container_name
    local health_status
    local project_name="${COMPOSE_PROJECT_NAME:-my-stack}"
    
    case $service in
        "traefik")
            container_name="${project_name}-traefik"
            ;;
        "redis")
            container_name="${project_name}-redis"
            ;;
        "fast-api-app")
            container_name="${project_name}-fast-api-app"
            ;;
        "firecrawl-postgres")
            container_name="${project_name}-firecrawl-postgres"
            ;;
        "firecrawl-playwright")
            container_name="${project_name}-firecrawl-playwright"
            ;;
        "firecrawl-api")
            container_name="${project_name}-firecrawl-api"
            ;;
        "searxng")
            container_name="${project_name}-searxng"
            ;;
        "supabase-db")
            container_name="${project_name}-supabase-db"
            ;;
        "supabase-kong")
            container_name="${project_name}-supabase-kong"
            ;;
        "supabase-auth")
            container_name="${project_name}-supabase-auth"
            ;;
        "supabase-rest")
            container_name="${project_name}-supabase-rest"
            ;;
        "supabase-realtime")
            container_name="${project_name}-realtime-dev.supabase-realtime"
            ;;
        "supabase-storage")
            container_name="${project_name}-supabase-storage"
            ;;
        "supabase-studio")
            container_name="${project_name}-supabase-studio"
            ;;
        "supabase-analytics")
            container_name="${project_name}-supabase-analytics"
            ;;
        "supabase-vector")
            container_name="${project_name}-supabase-vector"
            ;;
        "supabase-pooler")
            container_name="${project_name}-supabase-pooler"
            ;;
        *)
            container_name="${project_name}-${service}"
            ;;
    esac
    
    # Check if container is running
    if ! docker ps --format "table {{.Names}}" | grep -q "^${container_name}$"; then
        log_error "Container $container_name is not running"
        return 1
    fi
    
    # Check container health if healthcheck is defined
    health_status=$(docker inspect --format='{{.State.Health.Status}}' $container_name 2>/dev/null || echo "none")
    if [ "$health_status" = "healthy" ]; then
        log_success "$service is healthy"
        return 0
    elif [ "$health_status" = "none" ]; then
        # No healthcheck defined, just check if it's running
        log_success "$service is running (no healthcheck defined)"
        return 0
    else
        log_warning "$service health status: $health_status"
        return 0  # Don't fail for unhealthy status, just warn
    fi
}

check_service_health_quiet() {
    local service=$1
    local container_name
    local health_status
    local project_name="${COMPOSE_PROJECT_NAME:-my-stack}"
    
    case $service in
        "traefik")
            container_name="${project_name}-traefik"
            ;;
        "redis")
            container_name="${project_name}-redis"
            ;;
        "fast-api-app")
            container_name="${project_name}-fast-api-app"
            ;;
        "firecrawl-postgres")
            container_name="${project_name}-firecrawl-postgres"
            ;;
        "firecrawl-playwright")
            container_name="${project_name}-firecrawl-playwright"
            ;;
        "firecrawl-api")
            container_name="${project_name}-firecrawl-api"
            ;;
        "searxng")
            container_name="${project_name}-searxng"
            ;;
        "supabase-db")
            container_name="${project_name}-supabase-db"
            ;;
        "supabase-kong")
            container_name="${project_name}-supabase-kong"
            ;;
        "supabase-auth")
            container_name="${project_name}-supabase-auth"
            ;;
        "supabase-rest")
            container_name="${project_name}-supabase-rest"
            ;;
        "supabase-realtime")
            container_name="${project_name}-realtime-dev.supabase-realtime"
            ;;
        "supabase-storage")
            container_name="${project_name}-supabase-storage"
            ;;
        "supabase-studio")
            container_name="${project_name}-supabase-studio"
            ;;
        "supabase-analytics")
            container_name="${project_name}-supabase-analytics"
            ;;
        "supabase-vector")
            container_name="${project_name}-supabase-vector"
            ;;
        "supabase-pooler")
            container_name="${project_name}-supabase-pooler"
            ;;
        *)
            container_name="${project_name}-${service}"
            ;;
    esac
    
    # Check if container is running
    if ! docker ps --format "table {{.Names}}" | grep -q "^${container_name}$"; then
        return 1
    fi
    
    # Check container health if healthcheck is defined
    health_status=$(docker inspect --format='{{.State.Health.Status}}' $container_name 2>/dev/null || echo "none")
    if [ "$health_status" = "healthy" ]; then
        return 0
    elif [ "$health_status" = "none" ]; then
        # No healthcheck defined, just check if it's running
        return 0
    else
        return 1
    fi
}

check_http_service() {
    local service=$1
    local url=${SERVICE_URLS[$service]}
    
    if [ -z "$url" ]; then
        return 0  # Skip if no URL defined
    fi
    
    log_info "Checking HTTP service $service at $url..."
    
    local count=0
    local max_attempts=$((HEALTH_CHECK_TIMEOUT / HEALTH_CHECK_INTERVAL))
    
    while [ $count -lt $max_attempts ]; do
        if curl -s -f -o /dev/null "$url" 2>/dev/null; then
            log_success "$service is responding at $url"
            return 0
        fi
        
        count=$((count + 1))
        if [ $count -lt $max_attempts ]; then
            log_info "Waiting for $service to be ready... ($count/$max_attempts)"
            sleep $HEALTH_CHECK_INTERVAL
        fi
    done
    
    log_error "$service is not responding at $url after $HEALTH_CHECK_TIMEOUT seconds"
    return 1
}

# =============================================================================
# Service Testing
# =============================================================================

test_traefik() {
    log_info "Testing Traefik dashboard..."
    if curl -s -f "http://traefik.localhost:8080/dashboard/" > /dev/null; then
        log_success "Traefik dashboard is accessible"
        return 0
    else
        log_error "Traefik dashboard is not accessible"
        return 1
    fi
}

test_firecrawl() {
    log_info "Testing Firecrawl API..."
    local response=$(curl -s -w "%{http_code}" "http://firecrawl.localhost" -o /dev/null 2>/dev/null)
    if [ "$response" = "200" ] || [ "$response" = "404" ]; then
        log_success "Firecrawl API is responding (HTTP $response)"
        return 0
    else
        log_error "Firecrawl API is not responding (HTTP $response)"
        return 1
    fi
}

test_searxng() {
    log_info "Testing SearXNG..."
    local response=$(curl -s -w "%{http_code}" "http://searxng.localhost" -o /dev/null 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_success "SearXNG is responding"
        return 0
    else
        log_error "SearXNG is not responding (HTTP $response)"
        return 1
    fi
}

test_redis() {
    log_info "Testing Redis connection..."
    local project_name="${COMPOSE_PROJECT_NAME:-my-stack}"
    if docker exec ${project_name}-redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
        log_success "Redis is responding"
        return 0
    else
        log_error "Redis is not responding"
        return 1
    fi
}

test_fast_api_app() {
    log_info "Testing FastAPI App..."
    local response=$(curl -s -w "%{http_code}" "http://fastapi.localhost" -o /dev/null 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_success "FastAPI App is responding"
        return 0
    else
        log_error "FastAPI App is not responding (HTTP $response)"
        return 1
    fi
}

test_postgres() {
    log_info "Testing PostgreSQL connection..."
    local project_name="${COMPOSE_PROJECT_NAME:-my-stack}"
    if docker exec ${project_name}-firecrawl-postgres pg_isready -U postgres 2>/dev/null | grep -q "accepting connections"; then
        log_success "PostgreSQL is accepting connections"
        return 0
    else
        log_error "PostgreSQL is not accepting connections"
        return 1
    fi
}

test_supabase_studio() {
    log_info "Testing Supabase Studio..."
    local response=$(curl -s -w "%{http_code}" "http://supabase.localhost" -o /dev/null 2>/dev/null)
    if [ "$response" = "200" ]; then
        log_success "Supabase Studio is responding"
        return 0
    else
        log_error "Supabase Studio is not responding (HTTP $response)"
        return 1
    fi
}

test_supabase_kong() {
    log_info "Testing Supabase Kong API Gateway..."
    local project_name="${COMPOSE_PROJECT_NAME:-my-stack}"
    if docker exec ${project_name}-supabase-kong curl -s -f http://localhost:8000/ > /dev/null 2>/dev/null; then
        log_success "Supabase Kong is responding"
        return 0
    else
        log_error "Supabase Kong is not responding"
        return 1
    fi
}

test_supabase_auth() {
    log_info "Testing Supabase Auth..."
    local project_name="${COMPOSE_PROJECT_NAME:-my-stack}"
    if docker exec ${project_name}-supabase-auth curl -s -f http://localhost:9999/health > /dev/null 2>/dev/null; then
        log_success "Supabase Auth is responding"
        return 0
    else
        log_error "Supabase Auth is not responding"
        return 1
    fi
}

test_supabase_db() {
    log_info "Testing Supabase Database connection..."
    local project_name="${COMPOSE_PROJECT_NAME:-my-stack}"
    if docker exec ${project_name}-supabase-db pg_isready -U postgres 2>/dev/null | grep -q "accepting connections"; then
        log_success "Supabase Database is accepting connections"
        return 0
    else
        log_error "Supabase Database is not accepting connections"
        return 1
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo "=============================================================================="
    echo "Hybrid Docker Compose Setup - Complete Runner ($ENVIRONMENT)"
    echo "=============================================================================="
    echo
    
    # Setup environment
    setup_environment
    echo
    
    # Handle different commands
    local compose_cmd=$(get_docker_compose_cmd)
    case "$COMMAND" in
        up)
            start_services
            echo
            
            # Wait for services to start and become healthy
            log_info "Waiting for services to start and become healthy..."
            echo
            
            local max_wait_time=120  # Maximum wait time in seconds
            local check_interval=5   # Check interval in seconds
            local elapsed=0
            
            # Wait for all services to become healthy
            while [ $elapsed -lt $max_wait_time ]; do
                local all_healthy=true
                local unhealthy_services=()
                
                for service in "${SERVICES[@]}"; do
                    if ! check_service_health_quiet "$service"; then
                        all_healthy=false
                        unhealthy_services+=("$service")
                    fi
                done
                
                if [ "$all_healthy" = true ]; then
                    log_success "All services are healthy!"
                    echo
                    break
                fi
                
                log_info "Waiting for services to become healthy... (${elapsed}s elapsed)"
                log_info "Unhealthy services: ${unhealthy_services[*]}"
                echo
                
                sleep $check_interval
                elapsed=$((elapsed + check_interval))
            done
            
            if [ $elapsed -ge $max_wait_time ]; then
                log_warning "Timeout waiting for services to become healthy. Proceeding with tests anyway..."
                echo
            fi
            
            # Check service health
            log_info "Checking service health..."
            echo
            
            local failed_services=()
            
            for service in "${SERVICES[@]}"; do
                if ! check_service_health "$service"; then
                    failed_services+=("$service")
                fi
            done
            
            echo
            
            # Test HTTP services
            log_info "Testing HTTP endpoints..."
            echo
            
            if ! test_traefik; then
                failed_services+=("traefik-http")
            fi
            
            if ! test_firecrawl; then
                failed_services+=("firecrawl-http")
            fi
            
            if ! test_searxng; then
                failed_services+=("searxng-http")
            fi

            if ! test_fast_api_app; then
                failed_services+=("fast-api-app-http")
            fi
            
            if ! test_redis; then
                failed_services+=("redis-test")
            fi
            
            if ! test_postgres; then
                failed_services+=("postgres-test")
            fi
            
            if ! test_supabase_studio; then
                failed_services+=("supabase-studio-test")
            fi
            
            if ! test_supabase_kong; then
                failed_services+=("supabase-kong-test")
            fi
            
            if ! test_supabase_auth; then
                failed_services+=("supabase-auth-test")
            fi
            
            if ! test_supabase_db; then
                failed_services+=("supabase-db-test")
            fi
            
            echo
            echo "=============================================================================="
            
            # Final status
            if [ ${#failed_services[@]} -eq 0 ]; then
                log_success "All services are running and accessible!"
                echo
                echo "Service URLs:"
                echo "  - Traefik Dashboard: http://traefik.${BASE_DOMAIN:-localhost}:8080"
                echo "  - FastAPI App: http://fastapi.${BASE_DOMAIN:-localhost}"
                echo "  - Firecrawl API: http://firecrawl.${BASE_DOMAIN:-localhost}"
                echo "  - SearXNG: http://searxng.${BASE_DOMAIN:-localhost}"
                echo "  - Supabase Studio: http://supabase.${BASE_DOMAIN:-localhost}"
                echo
                echo "To view logs: $compose_cmd logs -f"
                echo "To stop services: $compose_cmd down"
            else
                log_error "Some services failed:"
                for service in "${failed_services[@]}"; do
                    echo "  - $service"
                done
                echo
                echo "To view logs: $compose_cmd logs"
                exit 1
            fi
            ;;
        down)
            log_info "Stopping all services..."
            $compose_cmd down --remove-orphans
            log_success "All services stopped."
            ;;
        logs)
            log_info "Following logs for all services..."
            $compose_cmd logs -f
            ;;
        *)
            log_info "Executing custom command: docker compose $COMMAND"
            $compose_cmd $COMMAND
            ;;
    esac
    
    echo "=============================================================================="
}

# Show usage information
show_usage() {
    echo "Usage: $0 [dev|prod] [command]"
    echo ""
    echo "Environments:"
    echo "  dev     Development environment (default)"
    echo "  prod    Production environment"
    echo ""
    echo "Commands:"
    echo "  up      Start services (default)"
    echo "  down    Stop and remove services"
    echo "  logs    View and follow logs"
    echo "  *       Pass any other command to docker compose"
    echo ""
    echo "Examples:"
    echo "  $0              # Start dev environment"
    echo "  $0 prod up      # Start prod environment"
    echo "  $0 dev logs     # View logs for dev environment"
    echo "  $0 prod pull    # Pull latest images for prod"
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Run main function
main "$@"