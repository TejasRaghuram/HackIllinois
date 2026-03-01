# Autonomous 911 Dispatch Agent - API Documentation

Welcome to the HackIllinois Autonomous 911 Dispatch Agent API. This backend service, built with FastAPI, handles live telephony streaming, AI interactions, and real-time dashboard updates.

## 📡 API Endpoints

### Data & State

| Method | Endpoint | Description | Response Type |
|--------|----------|-------------|---------------|
| `GET` | `/database` | Returns the entire sessions database as a JSON object, keyed by `session_id`. Each record includes the caller details, extracted location, required actions, real-time parsed transcripts, and a boolean `is_active` flag. | `application/json` |
| `WS` | `/database-stream` | A WebSocket endpoint that livestreams the exact same JSON payload as `/database`, pushing updates once every second. Perfect for live-updating UI dashboards without polling. | `WebSocket (JSON)` |

---

### Telephony & Streaming

| Method | Endpoint | Description | Response Type |
|--------|----------|-------------|---------------|
| `GET`/`POST` | `/voice` | Webhook URL for Twilio. Initiates the session and connects the incoming phone call to our WebSocket media stream. | `application/xml` (TwiML) |
| `WS` | `/media-stream` | Bidirectional WebSocket proxy that handles raw µ-law audio streaming between the Twilio voice call and the Gemini Live API model. | `WebSocket` |

---

### Dashboards & Monitoring

| Method | Endpoint | Description | Response Type |
|--------|----------|-------------|---------------|
| `GET` | `/sessions` | Live HTML dashboard displaying all active and past 911 sessions. Auto-refreshes every 3 seconds to show real-time transcripts, caller info, and agent dispatch decisions. | `text/html` |
| `GET` | `/logs` | Live HTML dashboard displaying the last 1000 backend logs. Auto-refreshes every 2 seconds. Useful for monitoring WebSocket and AI state. | `text/html` |
| `GET` | `/logs/clear` | Clears the current in-memory log history buffer and redirects back to the `/logs` page. | `HTTP 302 Redirect` |

---

### Diagnostics

| Method | Endpoint | Description | Response Type |
|--------|----------|-------------|---------------|
| `GET` | `/hello` | Basic health check/test endpoint. | `application/json` |

## 🗄️ Database Response Format (`GET /database`)

Example response from `/database`:

```json
{
  "unknown_1710000000": {
    "session_id": "unknown_1710000000",
    "phone_number": "+1234567890",
    "caller_name": "John Doe",
    "address": "123 Main St",
    "transcript": [
      { "role": "agent", "text": "911, what is your emergency?" },
      { "role": "caller", "text": "There's a fire in my kitchen!" }
    ],
    "actions": ["dispatch_fire"],
    "created_at": "2026-02-28 12:00:00",
    "is_active": false
  }
}
```
