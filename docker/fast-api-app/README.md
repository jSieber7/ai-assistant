# FastAPI App

A simple FastAPI application with health check support.

## Features

- Health check endpoint at `/health`
- Root endpoint at `/`
- Support for both development and production modes

## Endpoints

### Root
- **URL**: `/`
- **Method**: GET
- **Response**: `{"message": "FastAPI app is running"}`

### Health Check
- **URL**: `/health`
- **Method**: GET
- **Response**: `{"status": "healthy"}`

## Development

To run the app in development mode with hot reload:

```bash
docker compose --env-file .env up --build
```

## Production

To run the app in production mode:

```bash
BUILD_MODE=production docker compose --env-file .env up --build
```

## Health Check

The app includes a health check endpoint that can be used by Docker's health check mechanism:
```bash
curl -f http://localhost:8000/health