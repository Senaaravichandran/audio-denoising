# AudioClarity Deployment Guide

This guide covers various deployment methods for AudioClarity, from local development to production environments.

## 🚀 Quick Start Deployment

### Method 1: Automated Script (Recommended)

**Windows:**
```cmd
# Simply run the automated setup script
start.bat
```

**Linux/macOS:**
```bash
# Make script executable and run
chmod +x start.sh
./start.sh
```

The automated scripts will:
- Check system requirements
- Install dependencies automatically
- Configure the environment
- Start all services
- Open the application in your browser

### Method 2: Manual Installation

1. **Prerequisites**
   ```bash
   # Node.js 18+ and Python 3.9+
   node --version  # Should be 18+
   python --version  # Should be 3.9+
   ```

2. **Install Dependencies**
   ```bash
   npm install
   pip install -r ml/requirements.txt
   ```

3. **Environment Setup**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Add your Groq API key
   echo "GROQ_API_KEY=your_groq_api_key_here" >> .env
   ```

4. **Start Services**
   ```bash
   npm run dev
   ```

## ☁️ Cloud Deployment

### AWS EC2 Deployment

1. **Launch EC2 Instance**
   - Choose Ubuntu 22.04 LTS
   - Instance type: t3.medium or larger
   - Security groups: Allow ports 22, 80, 443, 3000

2. **Connect and Setup**
   ```bash
   # Connect to instance
   ssh -i your-key.pem ubuntu@your-instance-ip
   
   # Install Node.js
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   
   # Install Python
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

3. **Deploy Application**
   ```bash
   # Clone repository
   git clone https://github.com/yourusername/AudioClarity.git
   cd AudioClarity
   
   # Setup environment
   echo "GROQ_API_KEY=your_api_key" > .env
   
   # Install dependencies
   npm install
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r ml/requirements.txt
   
   # Build and start
   npm run build
   npm start
   ```

### DigitalOcean App Platform

1. **Create App**
   - Connect GitHub repository
   - Choose "Web Service" type
   - Set build command: `npm run build`
   - Set run command: `npm start`

2. **Environment Variables**
   ```
   GROQ_API_KEY=your_groq_api_key_here
   NODE_ENV=production
   ```

3. **Resource Configuration**
   - CPU: 1 vCPU
   - Memory: 2 GB
   - Storage: 10 GB

### Heroku Deployment

1. **Setup Heroku CLI**
   ```bash
   # Install Heroku CLI and login
   heroku login
   
   # Create new app
   heroku create your-app-name
   ```

2. **Configure Buildpacks**
   ```bash
   # Add Node.js and Python buildpacks
   heroku buildpacks:add heroku/nodejs
   heroku buildpacks:add heroku/python
   ```

3. **Deploy**
   ```bash
   # Set environment variables
   heroku config:set GROQ_API_KEY=your_api_key
   
   # Deploy
   git push heroku main
   ```

## 🔧 Production Configuration

### Environment Variables

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
NODE_ENV=production
PORT=3000
DATABASE_URL=sqlite:///app/data/audioclarity.db
REDIS_URL=redis://localhost:6379
MAX_FILE_SIZE=50mb
UPLOAD_TIMEOUT=300000
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/yourdomain.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.key;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    # Large file upload support
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### SSL Setup with Let's Encrypt

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 Monitoring and Logging

### Application Monitoring

```bash
# Health check endpoint
curl http://localhost:3000/api/health

# Metrics endpoint
curl http://localhost:3000/api/metrics
```

### Log Management

```bash
# Application logs (if using PM2)
pm2 logs audioclarity

# System logs
journalctl -u audioclarity -f

# Direct log files
tail -f logs/app.log

# Log rotation
sudo logrotate -f /etc/logrotate.d/audioclarity
```

### Performance Monitoring

1. **Built-in Monitoring**
   ```bash
   # PM2 monitoring
   pm2 monit
   
   # System resource monitoring
   htop
   iostat -x 1
   ```

2. **Application Metrics**
   - Request count and duration
   - Audio processing time
   - Memory and CPU usage
   - Error rates

## 🔒 Security Best Practices

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable

# iptables
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
```

### Security Headers

```javascript
// Express.js security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
    },
  },
}));
```

### Rate Limiting

```javascript
// API rate limiting
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP'
});

app.use('/api', limiter);
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port 3000
   lsof -i :3000
   netstat -tulpn | grep 3000
   
   # Kill process
   sudo kill -9 PID
   ```

3. **Memory Issues**
   ```bash
   # Increase Node.js memory limit
   export NODE_OPTIONS="--max-old-space-size=4096"
   
   # Monitor memory usage
   htop
   ps aux | grep node
   ```

3. **Audio Processing Errors**
   ```bash
   # Check FFmpeg installation
   ffmpeg -version
   
   # Check Python dependencies
   pip list | grep torch
   pip list | grep torchaudio
   ```

4. **Database Connection Issues**
   ```bash
   # Check database file permissions
   ls -la data/
   
   # Reset database
   rm data/audioclarity.db
   npm run db:migrate
   ```

### Log Analysis

```bash
# Search for errors
grep -i "error" logs/app.log

# Monitor real-time logs
tail -f logs/app.log | grep -E "(ERROR|WARN)"

# Analyze request patterns
awk '{print $1}' access.log | sort | uniq -c | sort -nr
```

## 📈 Scaling Considerations

### Horizontal Scaling

1. **Load Balancer Setup**
   ```nginx
   upstream audioclarity_backend {
       server 127.0.0.1:3000;
       server 127.0.0.1:3001;
       server 127.0.0.1:3002;
   }
   ```

2. **Session Management**
   ```javascript
   // Use Redis for session storage
   app.use(session({
     store: new RedisStore({ client: redisClient }),
     secret: process.env.SESSION_SECRET,
     resave: false,
     saveUninitialized: false
   }));
   ```

### Vertical Scaling

- **CPU**: Minimum 2 cores for production
- **Memory**: 4GB recommended for concurrent processing
- **Storage**: SSD recommended for fast I/O operations
- **Network**: High bandwidth for large audio file uploads

## 🔄 Backup and Recovery

### Database Backup

```bash
# SQLite backup
cp data/audioclarity.db backups/audioclarity_$(date +%Y%m%d_%H%M%S).db

# PostgreSQL backup
pg_dump audioclarity > backups/audioclarity_$(date +%Y%m%d_%H%M%S).sql
```

### Application Backup

```bash
# Full application backup
tar -czf audioclarity_backup_$(date +%Y%m%d).tar.gz \
  --exclude=node_modules \
  --exclude=.git \
  --exclude=outputs \
  AudioClarity/
```

### Automated Backups

```bash
# Add to crontab
0 2 * * * /path/to/backup_script.sh
```

## 🌐 CDN and Performance

### CloudFlare Setup

1. **Add Domain to CloudFlare**
2. **Configure DNS Records**
   ```
   A record: @ -> your-server-ip
   CNAME: www -> yourdomain.com
   ```
3. **Enable Performance Features**
   - Auto Minify (CSS, JS, HTML)
   - Brotli compression
   - Browser cache TTL

### Performance Optimization

```javascript
// Compression middleware
app.use(compression({
  level: 6,
  threshold: 1000,
  filter: (req, res) => {
    return compression.filter(req, res);
  }
}));

// Static file caching
app.use(express.static('public', {
  maxAge: '1d',
  etag: true
}));
```

## 📞 Support and Maintenance

### Health Checks

```bash
# Application health
curl -f http://localhost:3000/api/health || echo "App down"

# Database health
sqlite3 data/audioclarity.db ".tables" || echo "DB issue"

# Disk space
df -h | awk '$5 > 80 {print "Low disk space: " $0}'
```

### Maintenance Tasks

1. **Weekly**
   - Check logs for errors
   - Monitor disk usage
   - Review performance metrics

2. **Monthly**
   - Update dependencies
   - Backup database
   - Security audit

3. **Quarterly**
   - Performance optimization review
   - Capacity planning
   - Documentation updates

---

For additional support, please refer to our [GitHub Issues](https://github.com/yourusername/AudioClarity/issues) or [Discussions](https://github.com/yourusername/AudioClarity/discussions).
