# AWS Service to Semantic Color Mapping

To maintain visual consistency across infrastructure diagrams, use the following mapping to assign components to the design system's semantic categories.

| Service / Component | Semantic Category | Fill (rgba) | Stroke (Hex) |
| :--- | :--- | :--- | :--- |
| **Frontend** (React, Vue, iOS/Android App) | Frontend | `rgba(8, 51, 68, 0.4)` | `#22d3ee` |
| **Compute** (ECS Fargate, Lambda, EC2, Node.js) | Backend | `rgba(6, 78, 59, 0.4)` | `#34d399` |
| **Persistence** (RDS, DynamoDB, Aurora) | Database | `rgba(76, 29, 149, 0.4)` | `#a78bfa` |
| **Caching** (ElastiCache, Redis, Memcached) | Database | `rgba(76, 29, 149, 0.4)` | `#a78bfa` |
| **Infrastructure** (CloudFront, S3, ALB, API Gateway) | AWS/Cloud | `rgba(120, 53, 15, 0.3)` | `#fbbf24` |
| **Auth/Security** (IAM, Auth0, Cognito, WAF) | Security | `rgba(136, 19, 55, 0.4)` | `#fb7185` |
| **Messaging** (SQS, SNS, Kafka, EventBridge) | Message Bus | `rgba(251, 146, 60, 0.3)` | `#fb923c` |
| **External** (Third-party APIs, External Users) | External | `rgba(30, 41, 59, 0.5)` | `#94a3b8` |

### Logic Notes
- **Storage vs Database**: S3 is typically mapped to `AWS/Cloud` as it is an object store/infrastructure piece, whereas RDS/DynamoDB are mapped to `Database`.
- **Compute**: Anything performing business logic (the "API layer") should be `Backend`.
- **Load Balancers**: Treated as `AWS/Cloud` infrastructure unless they are being highlighted as a primary security boundary.
