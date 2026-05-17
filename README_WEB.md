# LK20 Web Platform

This is the local-hosted web platform for the LK20 governed curriculum digital-twin system. It provides a simple, dependency-free dashboard to interact with the `lk20_main.py` gateway.

## Features
- **Local API Server**: Built entirely with Python's standard library (`http.server`).
- **Dependency-Free Frontend**: Pure Vanilla JS and CSS. No React, no Tailwind, no external CDNs.
- **Full Feature Set**: Session management, curriculum inspection, search, upload with multipart form handling, coverage analysis, and government benefit reporting.

## Security Note
**WARNING**: This server uses local development authentication. Roles are trusted without cryptographic validation in this interface. 
**Do not bind this server publicly** (e.g. `0.0.0.0`) until proper authentication and network ingress rules are added. By default, the server binds to `127.0.0.1` to prevent external access.

## How to Run

1. Open a terminal or PowerShell in the project directory (`C:\Users\ali_z\ANU AI\LK20`).
2. Start the local server:
   ```powershell
   python lk20_server.py
   ```
   (Optional: Use `--port 8080` to change the port).
3. Open your browser and navigate to:
   http://127.0.0.1:8000

## Recommended First Sequence
If you are starting fresh without a network:
1. Go to the **Overview** tab and click `Init Project`.
2. Go to the **Login** tab, select the `admin` role and click `Login`.
3. Go back to **Overview** and click `Create Network`.
4. Go to **Login**, switch to `teacher` role and click `Login`.
5. Go to **Teacher Upload** and upload a test curriculum document.
6. Go to **Inspect** to view grade `G5` or subject `NOR`.
7. Go to **Coverage** to check curriculum coverage and gaps.

## Troubleshooting
- If you see `Data directory not found. Auto-initializing project...` in the console on the first run, the system has automatically initialized the folders.
- The **JSON Output** tab always shows the raw response from your last API call. Use it to debug any `false` ok statuses.
- Review the PowerShell console for tracebacks if an unexpected 500 error occurs.
