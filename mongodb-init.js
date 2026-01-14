// ============================================================================
// MongoDB Initialization Script for Document Verification API
// ============================================================================
// This script creates the necessary collections, indexes, and validation rules
// for the document verification inference logging system.
//
// Usage:
//   mongosh < mongodb-init.js
//   OR
//   Place this file in /docker-entrypoint-initdb.d/ for automatic initialization
// ============================================================================

// Switch to the database
db = db.getSiblingDB('document_verification');

print("=== Creating Document Verification Database ===");

// ============================================================================
// 1. Create Collections with Schema Validation
// ============================================================================

print("\n--- Creating inference_logs collection with schema validation ---");
db.createCollection("inference_logs", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["request_id", "timestamp", "api_version", "input", "environment", "response", "performance"],
      properties: {
        request_id: {
          bsonType: "string",
          description: "Unique UUID for each request"
        },
        timestamp: {
          bsonType: "date",
          description: "Request timestamp"
        },
        api_version: {
          bsonType: "string",
          description: "API version for tracking changes"
        },
        input: {
          bsonType: "object",
          required: ["image_hash", "image_size_bytes", "image_dimensions", "threshold_binary"],
          properties: {
            image_hash: {
              bsonType: "string",
              description: "SHA256 hash of decoded image"
            },
            image_size_bytes: {
              bsonType: "int",
              description: "Size of decoded image in bytes"
            },
            image_dimensions: {
              bsonType: "object",
              required: ["width", "height"],
              properties: {
                width: { bsonType: "int" },
                height: { bsonType: "int" }
              }
            },
            image_format: {
              bsonType: "string",
              description: "Image format (JPEG, PNG, etc.)"
            },
            threshold_binary: {
              bsonType: "double",
              description: "Classifier threshold parameter"
            }
          }
        },
        environment: {
          bsonType: "object",
          required: ["device", "hostname"],
          properties: {
            device: {
              bsonType: "string",
              enum: ["cuda", "cpu"],
              description: "Compute device used"
            },
            hostname: {
              bsonType: "string",
              description: "Server hostname"
            },
            container_id: {
              bsonType: ["string", "null"],
              description: "Docker container ID if applicable"
            },
            gpu_name: {
              bsonType: ["string", "null"],
              description: "GPU name if CUDA is used"
            },
            model_versions: {
              bsonType: "object",
              description: "Versions/names of all models used"
            }
          }
        },
        response: {
          bsonType: "object",
          required: ["timestamp", "ok", "decision_factors", "http_status"],
          properties: {
            ok: {
              bsonType: "bool",
              description: "Overall verification decision"
            },
            http_status: {
              bsonType: "int",
              description: "HTTP response status code"
            }
          }
        },
        performance: {
          bsonType: "object",
          required: ["total_duration_ms"],
          properties: {
            total_duration_ms: {
              bsonType: "double",
              description: "Total request processing time in milliseconds"
            }
          }
        },
        has_errors: {
          bsonType: "bool",
          description: "Whether any stage encountered errors"
        },
        error_stages: {
          bsonType: "array",
          items: { bsonType: "string" },
          description: "List of stages that encountered errors"
        }
      }
    }
  },
  validationLevel: "moderate",
  validationAction: "warn"
});

print("✓ inference_logs collection created");

// ============================================================================
// 2. Create model_registry collection (for tracking model versions)
// ============================================================================

print("\n--- Creating model_registry collection ---");
db.createCollection("model_registry", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["model_name", "model_type", "version", "registered_at"],
      properties: {
        model_name: {
          bsonType: "string",
          description: "Unique model identifier"
        },
        model_type: {
          bsonType: "string",
          enum: ["classifier", "ocr", "face_detector"],
          description: "Type of model"
        },
        version: {
          bsonType: "string",
          description: "Model version or checkpoint name"
        },
        registered_at: {
          bsonType: "date",
          description: "When this model was registered"
        },
        performance_baseline: {
          bsonType: "object",
          description: "Expected performance metrics"
        },
        metadata: {
          bsonType: "object",
          description: "Additional model information"
        }
      }
    }
  }
});

print("✓ model_registry collection created");

// ============================================================================
// 3. Create performance_metrics collection (for aggregated stats)
// ============================================================================

print("\n--- Creating performance_metrics collection ---");
db.createCollection("performance_metrics", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["metric_date", "metric_type", "aggregation_level"],
      properties: {
        metric_date: {
          bsonType: "date",
          description: "Date for this metric aggregation"
        },
        metric_type: {
          bsonType: "string",
          enum: ["daily", "hourly", "weekly"],
          description: "Aggregation time window"
        },
        aggregation_level: {
          bsonType: "string",
          enum: ["overall", "stage", "model"],
          description: "Level of aggregation"
        },
        total_requests: {
          bsonType: "int",
          description: "Number of requests in this period"
        },
        success_rate: {
          bsonType: "double",
          description: "Percentage of successful requests"
        },
        avg_duration_ms: {
          bsonType: "double",
          description: "Average processing time"
        },
        p95_duration_ms: {
          bsonType: "double",
          description: "95th percentile duration"
        },
        p99_duration_ms: {
          bsonType: "double",
          description: "99th percentile duration"
        }
      }
    }
  }
});

print("✓ performance_metrics collection created");

// ============================================================================
// 4. Create Indexes for Query Performance
// ============================================================================

print("\n--- Creating indexes on inference_logs ---");

// Primary query patterns
db.inference_logs.createIndex({ "request_id": 1 }, { unique: true });
print("  ✓ Unique index on request_id");

db.inference_logs.createIndex({ "timestamp": -1 });
print("  ✓ Index on timestamp (descending)");

db.inference_logs.createIndex({ "input.image_hash": 1 });
print("  ✓ Index on image_hash (for deduplication)");

db.inference_logs.createIndex({ "response.ok": 1, "timestamp": -1 });
print("  ✓ Compound index on response.ok and timestamp");

db.inference_logs.createIndex({ "environment.device": 1 });
print("  ✓ Index on device type");

db.inference_logs.createIndex({ "has_errors": 1, "timestamp": -1 });
print("  ✓ Index on has_errors for error tracking");

// Performance analysis indexes
db.inference_logs.createIndex({ "performance.total_duration_ms": -1 });
print("  ✓ Index on total_duration_ms");

db.inference_logs.createIndex({ 
  "binary_classifier.predictions.passed": 1, 
  "timestamp": -1 
});
print("  ✓ Index on binary_classifier.predictions.passed");

db.inference_logs.createIndex({ 
  "ocr_verification.marker_validation.is_valid_format": 1, 
  "timestamp": -1 
});
print("  ✓ Index on ocr_verification validity");

db.inference_logs.createIndex({ 
  "face_detection.detection_results.ok": 1, 
  "timestamp": -1 
});
print("  ✓ Index on face_detection.ok");

// Client tracking
db.inference_logs.createIndex({ "client_info.ip_address": 1 });
print("  ✓ Index on client_info.ip_address");

// TTL index for automatic data cleanup (optional - keeps last 90 days)
// Uncomment the following line to enable automatic deletion of old logs
// db.inference_logs.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 7776000 }); // 90 days
// print("  ✓ TTL index for automatic cleanup (90 days)");

print("\n--- Creating indexes on model_registry ---");
db.model_registry.createIndex({ "model_name": 1, "version": 1 }, { unique: true });
print("  ✓ Unique compound index on model_name and version");

db.model_registry.createIndex({ "registered_at": -1 });
print("  ✓ Index on registered_at");

print("\n--- Creating indexes on performance_metrics ---");
db.performance_metrics.createIndex({ 
  "metric_date": -1, 
  "metric_type": 1, 
  "aggregation_level": 1 
});
print("  ✓ Compound index on metric_date, metric_type, aggregation_level");

// ============================================================================
// 5. Insert Initial Model Registry Entries
// ============================================================================

print("\n--- Inserting initial model registry entries ---");

db.model_registry.insertMany([
  {
    model_name: "efficientnet_b0_binary",
    model_type: "classifier",
    version: "best_efficientnet_binary.pt",
    registered_at: new Date(),
    performance_baseline: {
      avg_inference_ms: 200,
      expected_accuracy: 0.95
    },
    metadata: {
      architecture: "EfficientNet-B0",
      num_classes: 2,
      input_size: [224, 224],
      normalization: "imagenet"
    }
  },
  {
    model_name: "paddleocr_english",
    model_type: "ocr",
    version: "2.7.0",
    registered_at: new Date(),
    performance_baseline: {
      avg_inference_ms: 750,
      expected_confidence: 0.85
    },
    metadata: {
      engine: "PaddleOCR",
      language: "en",
      use_angle_cls: true
    }
  },
  {
    model_name: "retinaface_resnet50",
    model_type: "face_detector",
    version: "resnet50_2020-07-20",
    registered_at: new Date(),
    performance_baseline: {
      avg_inference_ms: 300,
      expected_detection_rate: 0.98
    },
    metadata: {
      backbone: "ResNet-50",
      confidence_threshold: 0.5,
      max_size: 2048
    }
  }
]);

print("✓ Inserted 3 model registry entries");

// ============================================================================
// 6. Create Views for Common Queries (Optional)
// ============================================================================

print("\n--- Creating views for common analytics queries ---");

// View: Recent successful verifications
db.createView(
  "recent_successful_verifications",
  "inference_logs",
  [
    { $match: { "response.ok": true } },
    { $sort: { "timestamp": -1 } },
    { $limit: 100 },
    { $project: {
        request_id: 1,
        timestamp: 1,
        "performance.total_duration_ms": 1,
        "input.image_hash": 1,
        "response.ok": 1
      }
    }
  ]
);
print("✓ Created view: recent_successful_verifications");

// View: Failed verifications with reasons
db.createView(
  "failed_verifications",
  "inference_logs",
  [
    { $match: { "response.ok": false } },
    { $sort: { "timestamp": -1 } },
    { $limit: 100 },
    { $project: {
        request_id: 1,
        timestamp: 1,
        "response.decision_factors": 1,
        "binary_classifier.predictions.passed": 1,
        "ocr_verification.marker_validation.is_valid_format": 1,
        "face_detection.detection_results.ok": 1,
        "face_detection.detection_results.reason": 1
      }
    }
  ]
);
print("✓ Created view: failed_verifications");

// View: Performance statistics
db.createView(
  "performance_stats",
  "inference_logs",
  [
    {
      $group: {
        _id: "$environment.device",
        total_requests: { $sum: 1 },
        avg_duration_ms: { $avg: "$performance.total_duration_ms" },
        min_duration_ms: { $min: "$performance.total_duration_ms" },
        max_duration_ms: { $max: "$performance.total_duration_ms" },
        success_rate: {
          $avg: { $cond: ["$response.ok", 1, 0] }
        }
      }
    },
    { $sort: { total_requests: -1 } }
  ]
);
print("✓ Created view: performance_stats");

// ============================================================================
// 7. Verification and Summary
// ============================================================================

print("\n=== Database Initialization Complete ===");
print("\nCollections created:");
print("  • inference_logs (with schema validation)");
print("  • model_registry");
print("  • performance_metrics");

print("\nIndexes created:");
print("  • 11 indexes on inference_logs");
print("  • 2 indexes on model_registry");
print("  • 1 index on performance_metrics");

print("\nViews created:");
print("  • recent_successful_verifications");
print("  • failed_verifications");
print("  • performance_stats");

print("\nModel registry entries:");
print("  • EfficientNet-B0 Binary Classifier");
print("  • PaddleOCR English v2.7.0");
print("  • RetinaFace ResNet-50");

print("\n--- Database Statistics ---");
print("Collections:");
db.getCollectionNames().forEach(function(col) {
  var stats = db[col].stats();
  print("  • " + col + ": " + stats.count + " documents");
});

print("\n✓ Document Verification Database is ready!");
print("\nConnection string: mongodb://localhost:27017/document_verification");
print("To view logs: db.inference_logs.find().sort({timestamp: -1}).limit(10).pretty()");
