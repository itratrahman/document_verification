# MongoDB Credentials Folder

This folder contains MongoDB authentication credentials for the Document Verification API.

## ⚠️ SECURITY WARNING

**Never commit actual credentials to version control!**

This folder should contain:
- `mongodb_credentials.json` - Your actual credentials (git-ignored)
- `mongodb_credentials.json.example` - Template file (safe to commit)

## Setup Instructions

1. **Copy the example file:**
   ```bash
   cp mongodb_credentials.json.example mongodb_credentials.json
   ```

2. **Edit `mongodb_credentials.json` with your actual credentials:**
   ```json
   {
     "username": "your_actual_username",
     "password": "your_actual_password",
     "host": "localhost",
     "port": "27017",
     "database": "document_verification"
   }
   ```

3. **Verify `.gitignore` includes:**
   ```
   cred/mongodb_credentials.json
   ```

## File Format

```json
{
  "username": "admin",              // MongoDB username
  "password": "secure_password",    // MongoDB password
  "host": "localhost",              // MongoDB host (default: localhost)
  "port": "27017",                  // MongoDB port (default: 27017)
  "database": "document_verification"  // Database name (optional)
}
```

## Usage

The API automatically loads credentials from `cred/mongodb_credentials.json` on startup:
- ✓ If file exists and is valid → connects with authentication
- ✓ If file is missing → attempts connection without authentication (for local dev)
- ✗ If file is invalid → logs error and continues without MongoDB

## Security Best Practices

1. ✓ **Restrict file permissions:**
   ```bash
   # Linux/Mac
   chmod 600 cred/mongodb_credentials.json
   
   # Windows (PowerShell)
   icacls cred\mongodb_credentials.json /inheritance:r /grant:r "$($env:USERNAME):(R,W)"
   ```

2. ✓ **Use strong passwords:**
   - Minimum 16 characters
   - Mix of uppercase, lowercase, numbers, symbols
   - Avoid dictionary words

3. ✓ **Rotate credentials regularly:**
   - Change passwords every 90 days
   - Update both MongoDB and the credentials file

4. ✓ **Use environment-specific credentials:**
   - Development: `mongodb_credentials.dev.json`
   - Production: `mongodb_credentials.prod.json`
   - Never reuse credentials across environments

5. ✓ **For production, consider:**
   - AWS Secrets Manager
   - Azure Key Vault
   - HashiCorp Vault
   - Docker secrets

## Troubleshooting

**Issue: "MongoDB credentials file not found"**
- Create `mongodb_credentials.json` from the example file
- Or run without authentication for local development

**Issue: "Authentication failed"**
- Verify username/password are correct
- Check MongoDB user has proper permissions
- Ensure special characters in password are valid JSON

**Issue: "Connection timeout"**
- Verify MongoDB is running
- Check host and port are correct
- Ensure firewall allows connection
