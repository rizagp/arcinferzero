# Web UI plan: MVP, motion, and the full version (2026-11-05)

The web UI (`webui/index.html`, one file served by `serverz`) grew feature by feature: chats, projects, files,
transcripts and meetings, live recording, voice, terminal, memory, MCP. This plan sorts what exists into an MVP (what
must always work, and work well) and the full version, and lists what was added on 2026-11-05.

## 1. MVP: what must always work

The MVP is the shortest path from "open the page" to "a good answer", on this computer and on a phone.

| # | Part | State |
|---|---|---|
| M1 | Chat: send, stream the answer, stop, retry, Markdown and code, thinking box | done |
| M2 | Chat list: new, search, rename, delete; kept on disk (`/v1/chats`) | done |
| M3 | Model: pick and load, see what is loaded and how much GPU memory is left | done |
| M4 | Attachments: images, documents (PDF, DOCX, XLSX, PPTX), audio | done |
| M5 | Settings that change answers: thinking, temperature, system prompt, theme | done |
| M6 | Works on a phone: one column, back buttons, touch targets | done (760 px layout) |
| M7 | **Reachable from outside, safely**: password, session, tunnel | **added 2026-11-05** (docs/remote-access.md) |
| M8 | **Feels responsive**: motion that shows what happened, a sign that the model is working before the first token | **added 2026-11-05** (section 2) |
| M9 | Installable as an app (home screen icon, its own window) | **added 2026-11-05** (`/manifest.webmanifest`, `/icon.svg`) |

Everything else is the full version: it builds on the MVP and must not slow it down.

## 2. Motion

The rules:

- **Short:** 120-300 ms, eased out.
- **Cheap:** only `transform` and `opacity`, so nothing is laid out again. The model shares the iGPU with the
  browser, and streaming already limits repaints to one every 120 ms or more.
- **Never endless, except while waiting:** the only looping animations are the waiting dots and the prompt
  progress sheen, and both end when the first token comes.
- **Off when the system asks:** `prefers-reduced-motion: reduce` removes every animation and transition.

| Where | What |
|---|---|
| New message | slides in from its side (yours from the right, the model's from the left), 260 ms. Opening a chat draws its messages at once, without animation (`thread.dataset.loading`) |
| Waiting for the first token | three bouncing dots in the empty answer, while its progress bar is there |
| Prompt progress | a moving sheen until the server reports how far it is, then the bar fills smoothly |
| Menus, dropdown lists | pop in from their corner, 120-140 ms |
| "Latest" button, attachment chips | pop in with a little spring |
| Jobs bar, notes, the empty-chat card | fade up |
| Settings / Files / Transcripts panels | slide, with the same easing as everything else |
| Buttons | a quick press (scale 0.9) |
| Phone | opening a chat or a transcript slides it in from the right; going back fades the list in |
| Sign-in page (remote) | the card rises in; a wrong password shakes it |

## 3. Full version

| # | Part | State |
|---|---|---|
| F1 | Projects (folders, AGENTS.md, memory, MCP per project) | done (QUEUE F32) |
| F2 | Transcripts, meetings, minutes, live recording, two tracks | done |
| F3 | Voice (Talk), text to speech | done; microphone test by the user open (QUEUE F24) |
| F4 | Terminal in the chat, file tools, web search / fetch / view | done |
| F5 | Remote access hardening: Cloudflare Access in front, optional | documented (docs/remote-access.md) |
| F6 | Remote: per-device session list with "sign this device out" | later |
| F7 | Remote: an API key for programs outside (Pi on another computer) | later. Today a remote request needs the browser session |
| F8 | Offline shell (service worker) so the installed app opens without the network and says the server is away | later |
| F9 | Push notification when a long answer or a transcript is done | later: needs a service worker and the browser's permission |
| F10 | Split `index.html` (5,300 lines) into modules once a build step is acceptable | later: one file keeps the server simple |

## 4. Tests

- **Server side:** a scratch script runs 43 checks against a test server with its own data folder, and the gate is
  checked by its effects. It sends requests with a remote Host header, as the tunnel does, and checks:
  - the login page;
  - 401 without a session;
  - the wrong-password delay;
  - the lockout;
  - the cookie flags;
  - origin checks;
  - the blocked tools;
  - the wildcard host;
  - `Cf-Ray` with a local Host;
  - logout;
  - after an independent review: bodies refused before the password, parallel guesses, a lockout per visitor,
    sessions ending when the host names change, and the local-only paths (projects, memory, MCP, unload).
- **Page:** checked in the browser at desktop and phone width: the motion, the waiting dots, opening a chat without
  animation, and the remote settings (save, status). The sign-in page was checked over HTTP only, not looked at in a
  browser through a real tunnel.
