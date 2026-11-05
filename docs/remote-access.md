# Remote access (the web UI from a phone or another place)

The server answers only this computer by default. Remote access lets you reach the web UI from anywhere through a
**Cloudflare Tunnel**. A tunnel opens no port on your router: `cloudflared` on this computer connects out to Cloudflare
and passes the requests to the server. Every remote request needs a password.

## Set it up

1. **Install cloudflared:** `winget install --id Cloudflare.cloudflared`
2. **In the web UI on this computer, open Settings > Remote access:**
   - **Host names:** the address the tunnel will have:
     - for a quick tunnel (no Cloudflare account): `*.trycloudflare.com`;
     - for a named tunnel on your own domain: its name, for example `chat.example.com`.
   - **Password:** at least 10 characters. Use a long one: the page is on the internet.
   - **Over the internet:** on.
   - **Save.**
3. **Start the tunnel:**
   - Quick tunnel: `.\scripts\remote-tunnel.ps1`. It prints an address `https://<words>.trycloudflare.com`, which
     changes every time the tunnel starts.
   - Named tunnel, with a fixed address on your domain:
     1. `cloudflared tunnel login`, once.
     2. `cloudflared tunnel create chat`.
     3. `cloudflared tunnel route dns chat chat.example.com`.
     4. Then `.\scripts\remote-tunnel.ps1 -Name chat` each time.
4. **Open the address on the phone** and sign in. To get an app icon, choose "Add to home screen" (Android) or
   "Share > Add to Home Screen" (iPhone).

## What is protected, and how

- **Password.** It is stored as PBKDF2-HMAC-SHA256 (200,000 rounds, random salt) in `remote.json` in the data
  folder, never as the password itself.
  - A wrong password waits one second.
  - After 8 wrong passwords within 15 minutes from one visitor (by the address Cloudflare reports), that visitor is
    refused for 15 minutes. After 40 from everyone together, every sign-in is refused for 15 minutes.
  - Attempts still being checked count as wrong until they finish, so guesses sent in parallel cannot pass the
    limit.
- **Before you sign in, the server reads almost nothing.** A remote request without a session may only fetch a page
  (the sign-in page) or send a small sign-in. Anything else is refused before its body is read.
- **Session.** Signing in sets a cookie that is `HttpOnly`, `Secure` and `SameSite=Strict`, valid for 30 days.
  - These end every session: a restart of the server, a new password, "Sign everyone out", turning remote access
    off, changing the host names, or turning "Full access remotely" on.
  - "Sign out" in Settings ends this device's session.
- **Other websites cannot use your session.** A remote request may only come from the server's own page
  (`Origin: https://<host>`). Together with `SameSite=Strict`, another site cannot send requests as you.
- **A remote session is not "this computer" unless you allow it.** By default it can chat, read and write its chats
  and files, use web search, and work with transcripts and recordings. It cannot use:
  - the terminal (`run_command`);
  - projects and their folders;
  - the memory;
  - MCP servers (their `mcp.json` holds tokens);
  - loading or unloading models by hand (a chat that names a model still loads it);
  - the conversation slots;
  - saving OCR text into a folder.

  "Full access remotely" in Settings turns all of this on. Only do that if the password is strong: a remote session
  could then run any command on this computer.
- **Remote access settings can only be changed on this computer.** A remote session cannot read them, so it never
  sees the password hash.
- **Unknown host names are refused.** A request that came through a tunnel or proxy but names another host is
  refused. Such a request carries Cloudflare's `Cf-Ray` / `Cf-Connecting-IP`, or the `X-Forwarded-*`, `Forwarded`,
  `X-Real-IP` or `CDN-Loop` headers other proxies add. A tunnel set to rewrite the Host header to `127.0.0.1` would
  otherwise pass as a local request.
  - Do not use cloudflared's `--http-host-header`.
  - Only Cloudflare Tunnel is supported. Another tunnel (ngrok, Tailscale Funnel, a local nginx) must keep the
    visitor's Host header.
- **Files the model wrote** (HTML, SVG) open in a sandbox, even at their own address (`/files/<id>`), so a page with
  scripts cannot act as the web UI.
- **Only the server's own pages are affected.** Remote access does not change the model API for programs on this
  computer (Pi, scripts). They keep working without a password.

For more protection, put **Cloudflare Access** (Zero Trust, free for small teams) in front of a named tunnel. Then
Cloudflare asks for your e-mail login before the request even reaches this computer, and the server's password
comes on top of that.

## What does not work remotely

- **The microphone and live recording.** These work only over `https://`, which the tunnel gives you. They also
  need the browser's permission on the phone.
- **Opening files with the system's programs.** The "open in Explorer" links and the folder browser are for this
  computer only.
- **Speed over a slow connection.** Long answers stream as they are written, so a slow connection only delays
  them.

## Turning it off

- Switch "Over the internet" off in Settings and save: every remote request is refused at once.
- Closing the tunnel (Ctrl+C in its window) also takes the address off the internet.
