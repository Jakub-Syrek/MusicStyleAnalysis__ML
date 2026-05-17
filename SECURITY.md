# Security Policy

## Supported Versions

Security fixes are applied to the latest minor release on the default branch.

| Version | Supported |
| ------- | --------- |
| 1.x     | Yes       |
| < 1.0   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in MusicStyleAnalysis__ML, please
report it privately. Do **not** open a public GitHub issue for security
problems.

- Email: **jakubvonsyrek@gmail.com**
- Subject line: `[SECURITY] MusicStyleAnalysis__ML - <short summary>`

Please include:

- A description of the vulnerability and its impact.
- Steps to reproduce, including any sample audio files or URLs (avoid
  attaching copyrighted material; a public URL is preferred).
- The affected version (commit SHA or release tag).
- Any suggested mitigation, if known.

You can expect an acknowledgement within **72 hours** and a status update
within **7 days**. Once a fix is released, the advisory will be published on
the GitHub repository and credited to the reporter (unless anonymity is
requested).

## Scope

In scope:

- Code in `src/` and the published Python package.
- CI/CD workflows in `.github/workflows/`.
- Dependency-related vulnerabilities that affect this project's default
  installation (`pip install -r requirements.txt`).

Out of scope:

- Vulnerabilities in upstream libraries reported only against those projects
  (please report those upstream).
- Issues that require an attacker to already control the local machine.
