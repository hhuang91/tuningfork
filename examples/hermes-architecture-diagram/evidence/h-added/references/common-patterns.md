# Common Architectural Patterns: SVG Visuals

This reference provides SVG logic for recurring architectural patterns to ensure visual consistency and reduce coordinate guessing.

## 1. CDN Asset Bypass (Direct Delivery)
**Scenario:** A CDN (CloudFront) serves static assets directly from an Object Store (S3), bypassing the API/Compute layer.
- **Visual:** Use a curved `path` with a dashed stroke to indicate a separate data plane from the primary request flow.
- **SVG Logic:** 
  - Start: Right edge of CDN (`x + width`, `y + height/2`)
  - Control Point: Mid-point of the diagram, shifted upwards (`centerX`, `y - 100`)
  - End: Left edge of Storage (`x`, `y + height/2`)
- **Example:** `<path d="M [CDN_R] [CDN_Y] Q [Mid_X] [Mid_Y] [S3_L] [S3_Y]" fill="none" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="5,5" marker-end="url(#arrowhead)"/>`

## 2. Cache-Aside / Read-Through
**Scenario:** API checks a Cache (Redis) before hitting the Database (RDS).
- **Visual:** A triangular flow between the API, the Cache, and the Database.
- **SVG Logic:**
  - API $\rightarrow$ Cache: Straight line.
  - Cache $\rightarrow$ API: Return line (or bi-directional).
  - API $\rightarrow$ DB: Straight line.
  - Cache $\rightarrow$ DB: (Optional) for write-through.
- **Tip:** Place the Cache vertically between the API and DB to create a clear "layer" visual.

## 3. External Auth Loop (Identity Provider)
**Scenario:** User redirects to an external IDP (Auth0/Okta), authenticates, and returns with a token.
- **Visual:** A curved loop that exits the main flow and returns to the User/Client.
- **SVG Logic:**
  - User $\rightarrow$ IDP: Straight vertical or diagonal line.
  - IDP $\rightarrow$ User: Curved `path` returning to the User component.
  - User $\rightarrow$ API: Line carrying the resulting JWT.
- **Example:** `<path d="M [IDP_R] [IDP_Y] Q [Mid_X] [Mid_Y] [User_R] [User_Y]" ... />`

## 4. Database Cluster / Replica
**Scenario:** Primary DB with one or more Read Replicas.
- **Visual:** Primary DB at the top, Replicas grouped below it.
- **SVG Logic:** Use a light-colored boundary box to group the cluster, with a one-way arrow from Primary $\rightarrow$ Replicas to indicate replication.
