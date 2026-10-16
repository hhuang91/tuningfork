# Layout Patterns for Architecture Diagrams

When designing SVG architecture diagrams, using a consistent coordinate grid prevents overlap and ensures visual balance.

## 1. Linear Flow (The "Standard Web Stack")
Best for: Request-Response flows (Client -> API -> DB).
- **Direction:** Left-to-Right.
- **X-Axis Anchors:**
  - **External/Clients:** `x=50`
  - **Edge/CDN:** `x=220`
  - **Gateway/LB:** `x=400`
  - **Application/API:** `x=600`
  - **Persistence/Storage:** `x=800`
- **Y-Axis Anchors:**
  - **Primary Path:** `y=300` (Center line)
  - **Secondary/Supporting Services:** `y=200` (Above) or `y=400` (Below)
- **Sizing:**
  - **Component Width:** `100px` to `120px`
  - **Component Height:** `60px`
  - **Horizontal Gap:** `~80px`
  - **Vertical Gap:** `~40px`

## 2. Tiered Hub (The "Center-Out")
Best for: Event-driven architectures or central message buses.
- **Center:** Message Bus/Event Hub at `x=500, y=300`.
- **Periphery:** Producers to the left (`x=200`), Consumers to the right (`x=800`), and Management/Monitoring above/below.

## 3. Coordinate Calculation Tips
- **Centering Text:** For a component at `x` with `width`, set `text-anchor="middle"` and `x = x + (width / 2)`.
- **Arrow Alignment:** To hit the center of a component's edge:
  - **Right Edge:** `x2 = x + width`
  - **Left Edge:** `x1 = x + width` (of previous)
  - **Y-Center:** `y = y + (height / 2)`
- **Boundary Padding:** Give boundaries (`rect`) at least `20px` of padding around the internal components to avoid text clipping.
