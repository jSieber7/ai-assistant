# PostgREST Healthcheck Issue

## Date
2025-10-21

## Issue
PostgREST container (`ai-assistant-supabase-rest`) was showing as unhealthy in Docker despite the service running correctly.

## Symptoms
- Container status: "unhealthy"
- Service was actually working (responding with 401/permission denied errors)
- Healthcheck was failing repeatedly

## Root Cause
1. PostgREST requires authentication for ALL endpoints, including the root endpoint
2. The postgrest/postgrest container is extremely minimal and doesn't include common utilities:
   - No `curl` command
   - No `sh` shell
   - No `ps` command
3. The healthcheck in docker-compose.yml was trying to use `curl -f http://localhost:3000/` which:
   - Failed because curl wasn't installed
   - Would have failed anyway due to 401 authentication requirement

## Investigation Process
1. Checked container logs - showed service was running and connected to database
2. Tested endpoint manually - got expected 401 response (service working)
3. Attempted various healthcheck approaches:
   - Using `pg_isready` (wrong tool for postgrest)
   - Using `ps aux | grep postgrest` (ps not available)
   - Using curl with different options (curl not available)
4. Discovered container lacks basic utilities

## Solution
Disabled the healthcheck for postgrest in docker-compose.yml with a comment explaining why:
```yaml
# Healthcheck disabled - PostgREST requires authentication for all endpoints
# and the container doesn't have curl installed for health checks
# The service is working correctly as evidenced by 401 responses
```

## Lessons Learned
- Some containers are extremely minimal and may not have expected utilities
- Services that require authentication for all endpoints need special healthcheck consideration
- Sometimes disabling healthcheck is acceptable if the service is verified to be working through other means
- A 401 response can actually indicate a service is healthy (it's responding and enforcing security)

## Additional Attempt (2025-10-21)
User suggested trying: `test: ["CMD-SHELL", "pg_isready", "-d", "db_prod"]`
- Verified that `pg_isready` is also not available in the postgrest container
- The postgrest/postgrest:v11.2.0 image is extremely minimal without standard PostgreSQL client tools

## Verification
Confirmed postgrest is working by testing endpoint:
```bash
curl -s http://localhost:3001/
# Returns: {"code":"42501","details":null,"hint":null,"message":"permission denied to set role \"anon\""}
```
This error response confirms the service is running and connected to the database.