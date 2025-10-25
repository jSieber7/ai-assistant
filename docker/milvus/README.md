# Milvus Vector Database Setup

This directory contains the Docker Compose configuration for running Milvus v2.6.4 as a vector database service.

## Services

The setup includes three containers:

### 1. milvus-etcd
- **Purpose**: Metadata storage for Milvus
- **Image**: quay.io/coreos/etcd:v3.5.18
- **Ports**: Internal only (2379-2380/tcp)
- **Data Volume**: `./volumes/etcd:/etcd`

### 2. milvus-minio
- **Purpose**: Object storage for Milvus data
- **Image**: minio/minio:RELEASE.2024-12-18T13-15-44Z
- **Ports**: 
  - 9000 → MinIO API
  - 9001 → MinIO Console
- **Default Credentials**: 
  - Access Key: `minioadmin`
  - Secret Key: `minioadmin`
- **Data Volume**: `./volumes/minio:/minio_data`

### 3. milvus-standalone
- **Purpose**: Main Milvus vector database
- **Image**: milvusdb/milvus:v2.6.4
- **Ports**:
  - 19530 → Milvus gRPC API
  - 9091 → Milvus WebUI and Health Check
- **Data Volume**: `./volumes/milvus:/var/lib/milvus`

## Access Information

### Milvus WebUI (via Traefik)
- **URL**: http://milvus.localhost/webui/
- **Purpose**: Monitor and manage your Milvus instance
- **Status**: Available after all containers are healthy
- **Note**: Routed through Traefik reverse proxy

### MinIO Console (via Traefik)
- **URL**: http://minio.localhost/console
- **Username**: minioadmin
- **Password**: minioadmin
- **Purpose**: Object storage management for Milvus
- **Note**: Routed through Traefik reverse proxy

### Milvus gRPC API
- **Port**: 19530 (direct access)
- **Purpose**: Direct API access for applications
- **Note**: Exposed directly for performance (not through Traefik)

## Traefik Integration

The Milvus stack is now integrated with Traefik reverse proxy for secure and organized access:

- **Milvus WebUI**: Accessible at `http://milvus.localhost/webui/`
- **MinIO Console**: Accessible at `http://minio.localhost/console`
- **Load Balancing**: Automatic load balancing and health checks
- **SSL Ready**: Configuration prepared for SSL certificates (commented out for development)

### Direct Access (Alternative)
If you need direct access without Traefik:
- **Milvus WebUI**: http://127.0.0.1:9091/webui/ (if port exposed)
- **MinIO Console**: http://127.0.0.1:9001 (if port exposed)

## Management Commands

### Start Services
```bash
cd docker/milvus
docker compose up -d
```

### Check Status
```bash
cd docker/milvus
docker compose ps
```

### View Logs
```bash
cd docker/milvus
docker compose logs -f
```

### Stop Services
```bash
cd docker/milvus
docker compose down
```

### Stop Services and Remove Data
```bash
cd docker/milvus
docker compose down
sudo rm -rf volumes
```

## Data Persistence

All data is persisted in local volumes:
- `./volumes/etcd` - etcd metadata
- `./volumes/minio` - MinIO object storage
- `./volumes/milvus` - Milvus vector data

## Configuration

To modify Milvus configuration, you can access the milvus-standalone container:

```bash
docker exec -it milvus-standalone bash
```

Then edit `/milvus/configs/user.yaml` to override default settings and restart the container:

```bash
docker restart milvus-standalone
```

## Version Information

- Milvus Version: v2.6.4
- Installation Date: 2025-10-25
- Docker Compose Configuration: Downloaded from official Milvus releases

## Network

All containers are connected to a dedicated Docker network named `milvus` for internal communication.