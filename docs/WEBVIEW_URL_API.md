# Webview URL API – Integration Guide

This document describes how to integrate with the **Webview URL** endpoint. Use it to obtain a signed URL that loads the app frontend in a webview, with the user identified by phone number via an embedded JWT.

**Audience:** App teams (mobile or desktop) that need to open the FE in a webview. Written to be client-agnostic; adapt the steps to your stack (e.g. .NET, Kotlin, Swift).

---

## 1. Overview

| Item | Description |
|------|-------------|
| **Purpose** | Get a single URL that the app can open in a webview. The URL includes a JWT so the frontend knows the user (phone number). |
| **Method** | `GET` |
| **Path** | `/api/auth/webview-url` |
| **Auth** | FCM (Firebase Cloud Messaging) device token, sent in a header. |
| **Input** | User’s phone number (query parameter). |
| **Output** | JSON with a `url` field: the FE base URL with `token={jwt}` added as a query parameter. |

---

## 2. Base URL and Endpoint

- **Base URL:** Your API base (e.g. `https://api.example.com`). Confirm with the backend team.
- **Full endpoint:** `{BaseURL}/api/auth/webview-url`

Example: `https://api.example.com/api/auth/webview-url`

---

## 3. Authentication (FCM Token)

The endpoint is protected by **FCM token** authentication.

### 3.1 Obtaining the FCM token

- On the **device**, obtain the current FCM device token using your platform’s Firebase SDK (e.g. after registration or token refresh).
- Store this token in your app and use it when calling this API. The backend validates it with Firebase (dry-run) and rejects invalid or expired tokens.

### 3.2 Sending the FCM token

Send the FCM token in **one** of these ways (both are supported):

| Option | Header name | Value |
|--------|-------------|--------|
| **A** | `Authorization` | `Bearer <fcm_token>` |
| **B** | `X-FCM-Token` | `<fcm_token>` (raw token, no "Bearer ") |

- Use a single value; do not send both headers with different tokens.
- No other auth (e.g. API keys) is required for this endpoint.

---

## 4. Request

### 4.1 Method and path

```
GET /api/auth/webview-url
```

### 4.2 Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` **or** `X-FCM-Token` | Yes (one of them) | FCM device token (see §3). |
| `Content-Type` | Optional | Not required for GET; use `application/json` if you add a body (none expected). |
| Any other headers | Optional | Per your client (e.g. User-Agent, correlation IDs). |

### 4.3 Query parameters

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `phone` | Yes | string | User’s phone number. It is embedded in the issued JWT and used by the FE to identify the user. Normalize format as agreed with backend (e.g. E.164). |
| `user_type` | No | enum (`doctor` \| `farmer`) | User type embedded in the JWT. Defaults to `farmer` if omitted or empty. |

**Example:**  
`GET /api/auth/webview-url?phone=%2B919876543210`  
(phone = `+919876543210`, URL-encoded)

`GET /api/auth/webview-url?phone=%2B919876543210&user_type=doctor`

---

## 5. Response

### 5.1 Success (200 OK)

- **Body:** JSON object with at least:
  - `url` (string): Full URL to load in the webview (FE base URL + `token={jwt}` and any existing query params preserved).

**Example:**

```json
{
  "url": "https://app.example.com?token=<signed-jwt>"
}
```

- **Usage:** Open `url` in your webview. The frontend will read `token` from the query string and use it for its own auth/identity.

### 5.2 Response contract

- The API may include additional fields in the JSON in the future. Parse only the fields you need (e.g. `url`) and ignore others so your integration stays compatible.

---

## 6. Error Responses

| HTTP status | Meaning | Typical action |
|-------------|---------|----------------|
| **401 Unauthorized** | Missing or invalid FCM token (e.g. not sent, expired, or invalid). | Prompt re-login or refresh FCM token and retry. |
| **503 Service Unavailable** | Backend not configured for this feature (e.g. `APP_FE_URL` or Firebase not set). | Show a “service temporarily unavailable” message; retry later. |
| **500 Internal Server Error** | Server error (e.g. key/config issue). | Log, show generic error, retry with backoff. |

Error bodies are typically JSON with a `detail` (or similar) message; exact shape may vary. Rely on the HTTP status code for flow control; parse `detail` for logging or user-facing messages if present.

---

## 7. Integration Flow

1. **App start / after login**  
   Ensure you have:
   - A valid FCM device token (from Firebase SDK).
   - The user’s phone number (from your auth/session).

2. **Call the API**  
   - Method: `GET`.  
   - URL: `{BaseURL}/api/auth/webview-url?phone={phone}` (encode `phone`).  
   - Set either `Authorization: Bearer {fcm_token}` or `X-FCM-Token: {fcm_token}`.

3. **Handle response**  
   - **200:** Parse JSON, read `url`, open it in the webview.  
   - **401:** Refresh FCM token or re-authenticate user, then retry.  
   - **503 / 5xx:** Show “service unavailable” or generic error; retry later if appropriate.

4. **Webview**  
   Load the returned `url` in your webview. The FE will use the `token` query parameter for identity.

---

## 8. Example Request (cURL)

Replace placeholders with your base URL, FCM token, and phone number.

```bash
curl -X GET "https://api.example.com/api/auth/webview-url?phone=%2B919876543210" \
  -H "Authorization: Bearer YOUR_FCM_DEVICE_TOKEN"
```

Alternative header:

```bash
curl -X GET "https://api.example.com/api/auth/webview-url?phone=%2B919876543210" \
  -H "X-FCM-Token: YOUR_FCM_DEVICE_TOKEN"
```

---

## 9. Client Implementation Notes (Generic)

- **HTTP client:** Use your platform’s HTTP client (e.g. `HttpClient` in .NET, `URLSession` in iOS, `OkHttp` in Android). Prefer HTTPS and keep timeouts reasonable (e.g. 10–30 seconds).
- **Query string:** Add `phone` as a query parameter; ensure it is URL-encoded (especially `+`, spaces, etc.).
- **Headers:** Set exactly one of `Authorization: Bearer <fcm_token>` or `X-FCM-Token: <fcm_token>`.
- **Parsing:** Parse the JSON response and read the `url` field. Ignore unknown fields for forward compatibility.
- **Errors:** Branch on HTTP status (401, 503, 5xx) and handle as in §6. Do not treat non-2xx as success.
- **.NET:** Use `HttpClient`, set `DefaultRequestHeaders.Authorization` or a custom header for `X-FCM-Token`, build the request URI with `UriBuilder` or `QueryHelpers` for the `phone` parameter, and use `JsonSerializer` or `System.Text.Json` to deserialize the response. Handle `HttpRequestException` and check `response.IsSuccessStatusCode` and `response.StatusCode` for error handling.

---

## 10. JWT in the URL (For Frontend / Reference)

The `url` returned contains a query parameter `token` whose value is a JWT:

- **Signed with:** Backend’s RS256 private key (`jwt_private_key.pem`).
- **Typical claims:** `phone`, `sub` (same as phone), `user_type` (`doctor` or `farmer`, defaults to `farmer`), `iat`, `exp`, `aud`, `iss`. The FE can verify it with the backend’s public key and use `phone` (or `sub`) for identity and `user_type` to branch behaviour.
- **App responsibility:** The app only needs to open the returned URL in the webview; it does not need to parse or validate the JWT. JWT handling is between the backend and the frontend.

---

## 11. Summary Checklist for App Team

- [ ] Get FCM device token on the device (Firebase SDK).
- [ ] Get user phone number from your auth/session.
- [ ] Determine the `user_type` (`doctor` or `farmer`); omit it to default to `farmer`.
- [ ] Call `GET {BaseURL}/api/auth/webview-url?phone={encoded_phone}&user_type={doctor|farmer}`.
- [ ] Set `Authorization: Bearer {fcm_token}` or `X-FCM-Token: {fcm_token}`.
- [ ] On 200, parse JSON and open `url` in webview.
- [ ] On 401, refresh FCM token or re-auth and retry.
- [ ] On 503/5xx, show unavailable message and retry later if appropriate.

For base URL, environment (staging/production), and phone number format (e.g. E.164), confirm with the backend team.
