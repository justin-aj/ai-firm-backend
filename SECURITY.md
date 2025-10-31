# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please email the maintainers directly instead of opening a public issue.

## Security Best Practices

1. **Never commit `.env` files** - Keep your API keys and secrets in `.env` files which are gitignored
2. **Use environment variables** - All sensitive configuration should be in environment variables
3. **Keep dependencies updated** - Regularly update Python packages to get security patches
4. **CORS configuration** - Update CORS settings in production to only allow your frontend domains
5. **API rate limiting** - Consider implementing rate limiting for production use

## Environment Variables

Required environment variables:
- `GOOGLE_API_KEY` - Your Google Custom Search API key
- `GOOGLE_CX` - Your Google Custom Search Engine ID
- `DEBUG` - Set to `False` in production

## Production Checklist

Before deploying to production:
- [ ] Set `DEBUG=False` in `.env`
- [ ] Update CORS allowed origins in `main.py`
- [ ] Configure TrustedHostMiddleware with your domain
- [ ] Set up proper logging and monitoring
- [ ] Enable HTTPS/TLS
- [ ] Implement rate limiting
- [ ] Set up API key rotation schedule
- [ ] Review and limit API scopes
