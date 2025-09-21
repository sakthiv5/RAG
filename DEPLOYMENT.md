# RAG System Deployment Guide

This guide covers deployment options for the RAG system, from development to production environments.

## Deployment Options

### 1. Development Deployment (Local)

For development and testing:

```bash
# Start infrastructure services only
docker-compose up -d

# Run Flask app locally
python app.py
```

### 2. Full Stack Deployment (Docker)

For containerized deployment with all services:

```bash
# Build and start all services including Flask app
docker-compose --profile full-stack up -d --build

# Check service status
docker-compose ps
```

### 3. Production Deployment

For production environments:

```bash
# Use production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Monitor logs
docker-compose logs -f rag-app
```

## Configuration Management

### Environment Files

#### Development (.env)
```bash
OPENAI_API_KEY=sk-your-dev-key
FLASK_DEBUG=true
LOG_LEVEL=DEBUG
```

#### Production (.env)
```bash
OPENAI_API_KEY=sk-your-prod-key
FLASK_DEBUG=false
FLASK_SECRET_KEY=secure-production-key
LOG_LEVEL=WARNING
```

### Docker Profiles

The docker-compose.yml uses profiles to control service deployment:

- **Default**: Infrastructure services only (etcd, minio, milvus)
- **full-stack**: All services including Flask application

```bash
# Infrastructure only
docker-compose up -d

# Full stack
docker-compose --profile full-stack up -d
```

## Health Monitoring

### Health Check Endpoints

The application provides health check endpoints for monitoring:

#### Basic Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "rag-system",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Detailed Health Check
```bash
curl http://localhost:5000/health/detailed
```

Response:
```json
{
  "status": "healthy",
  "service": "rag-system",
  "timestamp": "2024-01-01T12:00:00Z",
  "components": {
    "rag_service": "healthy",
    "vector_store": "healthy",
    "environment": "healthy"
  }
}
```

### Docker Health Checks

Docker containers include built-in health checks:

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# View health check logs
docker inspect rag-flask-app --format='{{json .State.Health}}'
```

## Production Considerations

### Security

1. **API Keys**: Use secure key management
   ```bash
   # Use Docker secrets or external key management
   echo "sk-your-secure-key" | docker secret create openai_api_key -
   ```

2. **Network Security**: Configure firewalls and network policies
   ```bash
   # Example: Restrict Milvus access
   iptables -A INPUT -p tcp --dport 19530 -s 10.0.0.0/8 -j ACCEPT
   iptables -A INPUT -p tcp --dport 19530 -j DROP
   ```

3. **SSL/TLS**: Configure HTTPS for production
   ```nginx
   server {
       listen 443 ssl;
       server_name your-domain.com;
       
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       
       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Performance

1. **Resource Limits**: Configure appropriate resource limits
   ```yaml
   # In docker-compose.prod.yml
   deploy:
     resources:
       limits:
         memory: 2G
         cpus: '1.0'
   ```

2. **Scaling**: Use multiple Flask instances with load balancer
   ```bash
   # Scale Flask application
   docker-compose up -d --scale rag-app=3
   ```

3. **Caching**: Implement caching for frequently accessed data
   ```python
   # Example: Redis caching for query results
   import redis
   cache = redis.Redis(host='redis', port=6379, db=0)
   ```

### Monitoring

1. **Logging**: Configure centralized logging
   ```yaml
   # In docker-compose.prod.yml
   logging:
     driver: "json-file"
     options:
       max-size: "10m"
       max-file: "5"
   ```

2. **Metrics**: Implement application metrics
   ```python
   # Example: Prometheus metrics
   from prometheus_client import Counter, Histogram
   
   query_counter = Counter('rag_queries_total', 'Total queries processed')
   query_duration = Histogram('rag_query_duration_seconds', 'Query processing time')
   ```

3. **Alerting**: Set up monitoring alerts
   ```bash
   # Example: Health check monitoring
   */5 * * * * curl -f http://localhost:5000/health || echo "RAG system unhealthy" | mail admin@company.com
   ```

## Backup and Recovery

### Configuration Backup

```bash
# Backup configuration
python config_cli.py backup create production-$(date +%Y%m%d)

# List backups
python config_cli.py backup list

# Restore configuration
python config_cli.py backup restore backup_file.json
```

### Vector Database Backup

```bash
# Backup Milvus data
docker exec milvus-container milvus-backup create --collection pdf_rag_docs

# Restore Milvus data
docker exec milvus-container milvus-backup restore --collection pdf_rag_docs
```

### Document Backup

```bash
# Backup document data
tar -czf documents-backup-$(date +%Y%m%d).tar.gz data/

# Restore documents
tar -xzf documents-backup-20240101.tar.gz
```

## Troubleshooting Deployment

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :5000
   
   # Change port in environment
   export FLASK_PORT=5001
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats
   
   # Increase memory limits
   # Edit docker-compose.prod.yml
   ```

3. **Service Dependencies**
   ```bash
   # Check service startup order
   docker-compose logs milvus
   
   # Restart services in order
   docker-compose restart etcd minio milvus rag-app
   ```

### Log Analysis

```bash
# View application logs
docker-compose logs -f rag-app

# View infrastructure logs
docker-compose logs milvus etcd minio

# Search for errors
docker-compose logs rag-app 2>&1 | grep -i error
```

### Performance Debugging

```bash
# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Profile application
python -m cProfile -o profile.stats app.py

# Analyze query performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:5000/query
```

## Deployment Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] SSL certificates obtained (production)
- [ ] Firewall rules configured
- [ ] Backup strategy implemented
- [ ] Monitoring setup completed

### Deployment

- [ ] Services started successfully
- [ ] Health checks passing
- [ ] Application accessible
- [ ] Document processing working
- [ ] Query functionality tested

### Post-Deployment

- [ ] Performance monitoring active
- [ ] Log aggregation working
- [ ] Backup procedures tested
- [ ] Alert notifications configured
- [ ] Documentation updated

## Scaling Considerations

### Horizontal Scaling

```bash
# Scale Flask application
docker-compose up -d --scale rag-app=3

# Use load balancer (nginx example)
upstream rag_backend {
    server localhost:5000;
    server localhost:5001;
    server localhost:5002;
}
```

### Vertical Scaling

```yaml
# Increase resource limits
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
```

### Database Scaling

```bash
# Milvus cluster deployment
# Configure multiple Milvus nodes for high availability
```

## Maintenance

### Regular Tasks

1. **Log Rotation**: Ensure logs don't fill disk space
2. **Backup Verification**: Test backup and restore procedures
3. **Security Updates**: Keep Docker images and dependencies updated
4. **Performance Review**: Monitor and optimize query performance
5. **Configuration Cleanup**: Remove old configuration backups

### Update Procedures

```bash
# Update application
git pull origin main
docker-compose build rag-app
docker-compose up -d rag-app

# Update infrastructure
docker-compose pull
docker-compose up -d
```

This deployment guide provides comprehensive instructions for deploying the RAG system in various environments with proper monitoring, security, and maintenance considerations.