# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within FlashAudio, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email: security@flashvision.dev
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix & Disclosure**: Coordinated with reporter

## Scope

The following are in scope:
- Code execution vulnerabilities
- Model loading from untrusted sources (pickle/safetensors)
- Path traversal in config/checkpoint loading
- Audio file parsing vulnerabilities
- Dependency vulnerabilities

## Best Practices

When using FlashAudio:
- Only load models from trusted sources (HuggingFace Hub, verified checkpoints)
- Use `safetensors` format when possible (avoids pickle deserialization risks)
- Keep dependencies updated (`pip install --upgrade flashaudio`)
- Validate audio file inputs before processing
- Do not expose inference APIs to untrusted networks without authentication
