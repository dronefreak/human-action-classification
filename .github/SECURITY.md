# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.x     | :x:                |

**Note**: Version 1.x (TensorFlow 1.13) is no longer maintained due to end-of-life of TensorFlow 1.x.

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do NOT Open a Public Issue

Security vulnerabilities should not be disclosed publicly until a fix is available.

### 2. Report Privately

Please report security vulnerabilities by emailing:

**Email**: [kumaar324@gmail.com]

Or use GitHub's private vulnerability reporting:

- Go to the repository's Security tab
- Click "Report a vulnerability"
- Fill out the form with details

### 3. Include These Details

To help us understand and fix the issue quickly, please include:

- **Description**: Clear description of the vulnerability
- **Impact**: What could an attacker do?
- **Reproduction**: Step-by-step guide to reproduce
- **Affected versions**: Which versions are vulnerable?
- **Suggested fix**: If you have ideas (optional)
- **Your environment**: OS, Python version, etc.

### 4. What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 5 business days
- **Status updates**: Every 7 days until resolved
- **Fix timeline**: Critical issues within 7 days, others within 30 days

### 5. Coordinated Disclosure

Once we've patched the vulnerability:

1. We'll notify you before public disclosure
2. We'll release a security advisory
3. We'll credit you (unless you prefer to remain anonymous)

## Security Best Practices for Users

### Installation

```bash
# Always install from official sources
pip install -e .

# Verify package integrity
pip check
```

### Model Weights

```bash
# Only download weights from trusted sources
# - Official releases on GitHub
# - HuggingFace Hub
# - Official documentation links

# Verify checksums when provided
sha256sum weights/best.pth
```

### Docker

```bash
# Use official base images
# Keep images updated
docker pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Don't run as root in production
USER nonroot
```

### API Usage

```python
# Validate inputs
from pathlib import Path

def safe_predict(image_path: str):
    # Check file exists and is actually an image
    path = Path(image_path)
    if not path.exists():
        raise ValueError("File not found")

    if path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        raise ValueError("Invalid image format")

    # Proceed with prediction
    predictor.predict_image(str(path))
```

### Web Demo (Gradio)

```python
# Don't expose publicly without authentication
interface.launch(
    server_port=7860,
    share=False,  # Don't create public link in production
    auth=("username", "password")  # Add authentication
)
```

## Known Security Considerations

### 1. Model Poisoning

- **Risk**: Malicious model weights could execute arbitrary code
- **Mitigation**: Only use weights from trusted sources, verify checksums

### 2. Input Validation

- **Risk**: Malformed images could trigger vulnerabilities
- **Mitigation**: Use our built-in validation, set file size limits

### 3. Dependency Vulnerabilities

- **Risk**: Outdated dependencies may have known CVEs
- **Mitigation**: Run `pip audit` regularly, keep dependencies updated

### 4. Adversarial Attacks

- **Risk**: Carefully crafted images could fool the model
- **Mitigation**: This is a research challenge, not a security bug
  - For critical applications, use ensemble models
  - Add confidence thresholds
  - Implement human-in-the-loop verification

## Dependency Security

We regularly monitor dependencies for vulnerabilities:

```bash
# Check for known vulnerabilities
pip install pip-audit
pip-audit

# Update dependencies
pip install --upgrade -e ".[dev,demo,train]"
```

### Critical Dependencies

- **PyTorch**: Follow PyTorch security advisories
- **MediaPipe**: Google maintains this, generally secure
- **OpenCV**: Keep updated, has occasional CVEs
- **Pillow**: Image processing library, update regularly

## Security Updates

We publish security advisories via:

- GitHub Security Advisories
- Release notes with `[SECURITY]` prefix
- This SECURITY.md file

Subscribe to releases to get notifications:

1. Go to the repository
2. Click "Watch" → "Custom" → "Releases"

## Out of Scope

The following are **not** considered security vulnerabilities:

### 1. Model Accuracy Issues

- Misclassifications
- Low confidence predictions
- Bias in predictions
- _These are ML research issues, not security bugs_

### 2. Performance Issues

- Slow inference
- High memory usage
- _Performance bugs are tracked as regular issues_

### 3. Adversarial Examples

- Images specifically crafted to fool the model
- _This is an active research area, not a vulnerability_

### 4. Availability Issues

- API rate limiting
- Resource exhaustion from legitimate use
- _DOS from legitimate traffic isn't a security issue_

## Security-Related Configuration

### Recommended Production Settings

```python
# predictor.py - Production configuration
predictor = ActionPredictor(
    model_path='weights/verified_model.pth',  # Use verified weights
    device='cuda',
    use_pose_estimation=True
)

# Add input validation
import imghdr

def validate_image(image_path: str, max_size_mb: int = 10):
    """Validate image before processing."""
    path = Path(image_path)

    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"Image too large: {size_mb:.1f}MB")

    # Verify it's actually an image
    image_type = imghdr.what(path)
    if image_type not in ['jpeg', 'png', 'gif']:
        raise ValueError(f"Invalid image type: {image_type}")

    return True

# Use it before prediction
validate_image('user_upload.jpg')
result = predictor.predict_image('user_upload.jpg')
```

### Docker Security

```dockerfile
# Dockerfile - Security hardening
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Don't run as root
RUN useradd -m -u 1000 appuser
USER appuser

# Install only what's needed
COPY --chown=appuser:appuser . /app
WORKDIR /app

# Set resource limits
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Expose only necessary ports
EXPOSE 7860

# Use non-root user
USER appuser
CMD ["hac-demo", "--port", "7860"]
```

## Contact

For security issues: [kummar@324.com]
For general issues: [GitHub Issues](https://github.com/dronefreak/human-action-classification/issues)

## Credits

We appreciate security researchers who help keep this project safe:

- _Your name here_ - Reported [ISSUE-001]

## License

This security policy is licensed under CC0 1.0 Universal.

---

**Last Updated**: October 2025
**Next Review**: January 2026
