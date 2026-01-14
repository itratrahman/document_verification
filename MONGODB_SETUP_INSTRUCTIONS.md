# MongoDB Setup Instructions for Document Verification API

This guide provides step-by-step instructions to set up MongoDB for inference logging in both Windows (local development) and Docker Compose (production) environments.

---

## Table of Contents

1. [Windows Installation](#1-windows-installation-local-development)
2. [Docker Compose Setup](#2-docker-compose-setup-production)
3. [Database Initialization](#3-database-initialization)
4. [Testing the Setup](#4-testing-the-setup)
5. [Querying Logs](#5-querying-logs)
6. [Maintenance and Monitoring](#6-maintenance-and-monitoring)

---

## 1. Windows Installation (Local Development)

### Step 1.1: Download MongoDB Community Server

1. Visit the official MongoDB download page: https://www.mongodb.com/try/download/community
2. Select:
   - **Version**: 7.0.x (latest stable)
   - **Platform**: Windows
   - **Package**: MSI
3. Click **Download** and wait for the installer to download

### Step 1.2: Install MongoDB

1. Run the downloaded MSI installer
2. Choose **Complete** installation type
3. **Important**: Check the box for "Install MongoDB as a Service"
   - Service Name: `MongoDB`
   - Data Directory: `C:\Program Files\MongoDB\Server\7.0\data\`
   - Log Directory: `C:\Program Files\MongoDB\Server\7.0\log\`
4. **Optional**: Uncheck "Install MongoDB Compass" if you don't need the GUI (saves disk space)
5. Click **Install** and wait for completion
6. Click **Finish**

### Step 1.3: Add MongoDB to System PATH

1. Open **System Environment Variables**:
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Go to **Advanced** tab → **Environment Variables**
2. Under **System Variables**, find and select `Path`, click **Edit**
3. Click **New** and add:
   ```
   C:\Program Files\MongoDB\Server\7.0\bin
   ```
4. Click **OK** to save all dialogs

### Step 1.4: Verify MongoDB Installation

Open PowerShell and run:
```powershell
mongod --version
```

You should see output like:
```
db version v7.0.x
```

### Step 1.5: Start MongoDB Service

MongoDB should auto-start as a Windows service. Verify it's running:

```powershell
# Check service status
Get-Service MongoDB

# If not running, start it
Start-Service MongoDB
```

### Step 1.6: Install MongoDB Shell (mongosh)

The MongoDB Shell is needed to interact with the database:

1. Download from: https://www.mongodb.com/try/download/shell
2. Select:
   - **Platform**: Windows 64-bit (MSI)
   - **Package**: msi
3. Install the MSI package
4. Verify installation:
   ```powershell
   mongosh --version
   ```

---

## 2. Docker Compose Setup (Production)

### Step 2.1: Create Docker Compose Configuration

Create or update `docker-compose.yml` in your project root:

```yaml
version: '3.8'

services:
  # FastAPI Application
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      # MongoDB connection settings
      MONGODB_URI: mongodb://mongo:27017/
      MONGODB_DATABASE: document_verification
      MONGODB_TIMEOUT_MS: 5000
    depends_on:
      mongo:
        condition: service_healthy
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ./models:/app/models:ro
      - ./data:/app/data:ro
    networks:
      - app_network

  # MongoDB Service
  mongo:
    image: mongo:7.0
    container_name: document-verification-mongo
    restart: unless-stopped
    ports:
      - "27017:27017"
    environment:
      # Optional: Set authentication (recommended for production)
      # MONGO_INITDB_ROOT_USERNAME: admin
      # MONGO_INITDB_ROOT_PASSWORD: your_secure_password
      MONGO_INITDB_DATABASE: document_verification
    volumes:
      # Persistent data storage
      - mongodb_data:/data/db
      - mongodb_logs:/var/log/mongodb
      # Initialization script (automatically executed on first run)
      - ./mongodb-init.js:/docker-entrypoint-initdb.d/mongodb-init.js:ro
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    networks:
      - app_network

  # Optional: MongoDB Web UI (mongo-express)
  mongo-express:
    image: mongo-express:latest
    container_name: document-verification-mongo-express
    restart: unless-stopped
    ports:
      - "8081:8081"
    environment:
      ME_CONFIG_MONGODB_URL: mongodb://mongo:27017/
      ME_CONFIG_BASICAUTH_USERNAME: admin
      ME_CONFIG_BASICAUTH_PASSWORD: admin123
    depends_on:
      - mongo
    networks:
      - app_network

networks:
  app_network:
    driver: bridge

volumes:
  mongodb_data:
    driver: local
  mongodb_logs:
    driver: local
```

### Step 2.2: Update Dockerfile

Ensure your `Dockerfile` includes MongoDB environment variables:

```dockerfile
# ... existing Dockerfile content ...

# MongoDB connection settings (can be overridden by docker-compose)
ENV MONGODB_URI=mongodb://localhost:27017/
ENV MONGODB_DATABASE=document_verification
ENV MONGODB_TIMEOUT_MS=5000

# ... rest of Dockerfile ...
```

### Step 2.3: Update requirements.txt

Add MongoDB async driver to `requirements.txt`:

```txt
# ... existing requirements ...

# MongoDB async driver
motor>=3.3.0
pymongo>=4.6.0
```

### Step 2.4: Start Services

```powershell
# Build and start all services
docker-compose up -d --build

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f mongo
```

---

## 3. Database Initialization

### Option A: Automatic Initialization (Docker Only)

When using Docker Compose with the configuration above, the `mongodb-init.js` script runs automatically on the first container startup. This creates:

- ✓ `inference_logs` collection with schema validation
- ✓ `model_registry` collection
- ✓ `performance_metrics` collection
- ✓ All necessary indexes
- ✓ Initial model registry entries
- ✓ Analytics views

**No manual steps required!**

### Option B: Manual Initialization (Windows or Docker)

#### For Windows (Local MongoDB):

```powershell
# Navigate to your project directory
cd C:\Users\rahma\OneDrive\Desktop\document_verification

# Run the initialization script
mongosh < mongodb-init.js
```

#### For Docker (if automatic initialization didn't run):

```powershell
# Copy script into running container
docker cp mongodb-init.js document-verification-mongo:/tmp/init.js

# Execute script
docker exec -it document-verification-mongo mongosh /tmp/init.js
```

### Verify Initialization

Connect to MongoDB and check collections:

#### Windows:
```powershell
mongosh
```

#### Docker:
```powershell
docker exec -it document-verification-mongo mongosh
```

Then in the MongoDB shell:
```javascript
// Switch to database
use document_verification

// List collections
show collections
// Expected output: inference_logs, model_registry, performance_metrics

// Check indexes on inference_logs
db.inference_logs.getIndexes()
// Expected: 11+ indexes

// View model registry
db.model_registry.find().pretty()
// Expected: 3 model entries (EfficientNet, PaddleOCR, RetinaFace)
```

---

## 4. Testing the Setup

### Step 4.1: Install Python Dependencies

```powershell
# Install motor (async MongoDB driver)
pip install motor>=3.3.0 pymongo>=4.6.0
```

### Step 4.2: Set Environment Variables (Windows Only)

```powershell
# PowerShell
$env:MONGODB_URI = "mongodb://localhost:27017/"
$env:MONGODB_DATABASE = "document_verification"
$env:MONGODB_TIMEOUT_MS = "5000"
```

### Step 4.3: Start the API

#### Windows:
```powershell
python app.py
# OR
uvicorn app:app --reload
```

#### Docker:
```powershell
docker-compose up -d
```

### Step 4.4: Send Test Request

Create a test script `test_inference_log.py`:

```python
import requests
import base64

# Load a test image and encode to base64
with open("data/random_doc_images/passport/sample.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Send request to API
response = requests.post(
    "http://localhost:8000/verify",
    json={
        "image_base64": image_data,
        "thresh_binary": 0.5
    }
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

Run the test:
```powershell
python test_inference_log.py
```

### Step 4.5: Verify Log Was Created

#### Windows:
```powershell
mongosh
```

#### Docker:
```powershell
docker exec -it document-verification-mongo mongosh
```

Then:
```javascript
use document_verification

// Check for recent logs
db.inference_logs.find().sort({timestamp: -1}).limit(1).pretty()

// Should show your test request with all inference details
```

---

## 5. Querying Logs

### Common Queries

#### 5.1: View Recent Requests
```javascript
// Last 10 requests
db.inference_logs.find()
  .sort({timestamp: -1})
  .limit(10)
  .pretty()
```

#### 5.2: Find Failed Verifications
```javascript
// Requests where overall verification failed
db.inference_logs.find(
  { "response.ok": false }
).sort({timestamp: -1}).pretty()
```

#### 5.3: Search by Image Hash (Find Duplicates)
```javascript
// Find requests with same image
db.inference_logs.find(
  { "input.image_hash": "sha256-hash-here" }
).pretty()
```

#### 5.4: Performance Analysis
```javascript
// Requests slower than 2 seconds
db.inference_logs.find(
  { "performance.total_duration_ms": { $gt: 2000 } }
).sort({"performance.total_duration_ms": -1}).pretty()
```

#### 5.5: Stage-Specific Failures
```javascript
// Failed due to OCR issues
db.inference_logs.find({
  "ocr_verification.marker_validation.is_valid_format": false
}).count()

// Failed due to face detection
db.inference_logs.find({
  "face_detection.detection_results.ok": false
}).count()
```

#### 5.6: Aggregate Statistics
```javascript
// Success rate by date
db.inference_logs.aggregate([
  {
    $group: {
      _id: { $dateToString: { format: "%Y-%m-%d", date: "$timestamp" } },
      total_requests: { $sum: 1 },
      successful: {
        $sum: { $cond: ["$response.ok", 1, 0] }
      },
      avg_duration_ms: { $avg: "$performance.total_duration_ms" }
    }
  },
  {
    $project: {
      date: "$_id",
      total_requests: 1,
      success_rate: {
        $multiply: [{ $divide: ["$successful", "$total_requests"] }, 100]
      },
      avg_duration_ms: 1
    }
  },
  { $sort: { date: -1 } }
])
```

#### 5.7: Model Performance Comparison
```javascript
// Average inference time by device (CPU vs GPU)
db.inference_logs.aggregate([
  {
    $group: {
      _id: "$environment.device",
      avg_total_ms: { $avg: "$performance.total_duration_ms" },
      avg_classifier_ms: { $avg: "$binary_classifier.duration_ms" },
      avg_ocr_ms: { $avg: "$ocr_verification.duration_ms" },
      avg_face_ms: { $avg: "$face_detection.duration_ms" },
      count: { $sum: 1 }
    }
  }
])
```

---

## 6. Maintenance and Monitoring

### 6.1: Database Backup

#### Windows:
```powershell
# Create backup directory
mkdir C:\mongodb_backups

# Backup entire database
mongodump --db document_verification --out C:\mongodb_backups\$(Get-Date -Format "yyyy-MM-dd")

# Restore from backup
mongorestore --db document_verification C:\mongodb_backups\2026-01-13\document_verification
```

#### Docker:
```powershell
# Backup
docker exec document-verification-mongo mongodump --db document_verification --out /tmp/backup
docker cp document-verification-mongo:/tmp/backup ./mongodb_backup_$(Get-Date -Format "yyyy-MM-dd")

# Restore
docker cp ./mongodb_backup_2026-01-13 document-verification-mongo:/tmp/restore
docker exec document-verification-mongo mongorestore --db document_verification /tmp/restore/document_verification
```

### 6.2: Enable Automatic Data Cleanup (TTL)

To automatically delete logs older than 90 days, run this in MongoDB shell:

```javascript
use document_verification

// Create TTL index (deletes documents after 90 days)
db.inference_logs.createIndex(
  { "timestamp": 1 },
  { expireAfterSeconds: 7776000 }  // 90 days in seconds
)
```

### 6.3: Monitor Database Size

```javascript
// Check database statistics
db.stats()

// Check collection sizes
db.inference_logs.stats()

// Count documents
db.inference_logs.countDocuments()
```

### 6.4: Performance Optimization

```javascript
// Analyze slow queries
db.setProfilingLevel(1, { slowms: 100 })  // Log queries slower than 100ms

// View slow queries
db.system.profile.find().sort({ ts: -1 }).limit(5).pretty()

// Rebuild indexes (if performance degrades)
db.inference_logs.reIndex()
```

### 6.5: Access MongoDB Web UI (Docker Only)

If you enabled `mongo-express` in docker-compose:

1. Open browser: http://localhost:8081
2. Login:
   - Username: `admin`
   - Password: `admin123`
3. Navigate to `document_verification` database
4. Browse collections and execute queries

---

## Troubleshooting

### Issue: MongoDB service won't start (Windows)

**Solution:**
```powershell
# Check service status
Get-Service MongoDB

# View logs
Get-Content "C:\Program Files\MongoDB\Server\7.0\log\mongod.log" -Tail 50

# Restart service
Restart-Service MongoDB
```

### Issue: Connection refused (Docker)

**Solution:**
```powershell
# Check if container is running
docker ps

# Check container logs
docker logs document-verification-mongo

# Verify health check
docker inspect document-verification-mongo | Select-String -Pattern "Health"

# Restart container
docker-compose restart mongo
```

### Issue: API can't connect to MongoDB

**Solution:**
```powershell
# Check environment variables
docker exec document-verification-api env | Select-String -Pattern "MONGODB"

# Test connection from API container
docker exec -it document-verification-api python -c "from motor.motor_asyncio import AsyncIOMotorClient; import asyncio; asyncio.run(AsyncIOMotorClient('mongodb://mongo:27017/').admin.command('ping')); print('Connected!')"
```

### Issue: Initialization script didn't run

**Solution:**
```powershell
# Check if database already exists (script only runs on first startup)
docker exec -it document-verification-mongo mongosh --eval "show dbs"

# If document_verification exists, manually run init script
docker exec -it document-verification-mongo mongosh < /docker-entrypoint-initdb.d/mongodb-init.js
```

---

## Security Best Practices (Production)

### Enable Authentication

Edit `docker-compose.yml`:

```yaml
mongo:
  image: mongo:7.0
  environment:
    MONGO_INITDB_ROOT_USERNAME: admin
    MONGO_INITDB_ROOT_PASSWORD: your_secure_password_here
    MONGO_INITDB_DATABASE: document_verification
```

Update connection string in API service:

```yaml
api:
  environment:
    MONGODB_URI: mongodb://admin:your_secure_password_here@mongo:27017/
```

### Use Environment Files

Create `.env` file:
```env
MONGODB_ROOT_USERNAME=admin
MONGODB_ROOT_PASSWORD=your_secure_password
MONGODB_DATABASE=document_verification
```

Update `docker-compose.yml`:
```yaml
services:
  mongo:
    env_file: .env
  api:
    env_file: .env
```

**Important**: Add `.env` to `.gitignore`!

---

## Next Steps

1. ✓ MongoDB installed and running
2. ✓ Database initialized with collections and indexes
3. ✓ API connected to MongoDB
4. ✓ Test inference logged successfully

**You're ready to start logging inferences!**

For production deployment, consider:
- Enable MongoDB authentication
- Set up regular backups
- Configure TTL for automatic log cleanup
- Monitor database performance
- Set up replica sets for high availability

---

## Additional Resources

- MongoDB Documentation: https://docs.mongodb.com/
- Motor (Async Driver) Docs: https://motor.readthedocs.io/
- MongoDB Atlas (Cloud): https://www.mongodb.com/cloud/atlas
- Monitoring Tools: MongoDB Compass, Studio 3T
