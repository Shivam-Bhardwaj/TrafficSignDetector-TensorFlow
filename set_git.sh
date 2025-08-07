#!/bin/bash

# SECURITY WARNING: This script has been secured
# Original script set global git config which could affect other repositories
# and stored credentials insecurely

echo "🔒 SECURE GIT CONFIGURATION"
echo "This script helps you configure git safely for this repository only"
echo ""

# Validate we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Pull latest changes first
echo "📥 Updating repository..."
if ! git pull; then
    echo "⚠️  Warning: Could not pull latest changes. Continuing anyway..."
fi
echo ""

# Input validation function
validate_email() {
    local email=$1
    if [[ $email =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
        return 0
    else
        return 1
    fi
}

validate_url() {
    local url=$1
    if [[ $url =~ ^https?:// ]] || [[ $url =~ ^git@[a-zA-Z0-9.-]+: ]]; then
        return 0
    else
        return 1
    fi
}

validate_username() {
    local username=$1
    if [[ ${#username} -ge 1 && ${#username} -le 39 ]] && [[ $username =~ ^[a-zA-Z0-9._-]+$ ]]; then
        return 0
    else
        return 1
    fi
}

# Get and validate repository URL
while true; do
    read -p "🔗 Git repository URL (https://github.com/user/repo.git): " upstreamVar
    if [ -z "$upstreamVar" ]; then
        echo "❌ Repository URL cannot be empty"
        continue
    fi
    if validate_url "$upstreamVar"; then
        break
    else
        echo "❌ Invalid URL format. Use https:// or git@ format"
    fi
done

# Get and validate username
while true; do
    read -p "👤 Git Username: " userVar
    if [ -z "$userVar" ]; then
        echo "❌ Username cannot be empty"
        continue
    fi
    if validate_username "$userVar"; then
        break
    else
        echo "❌ Invalid username. Use only letters, numbers, dots, hyphens, and underscores"
    fi
done

# Get and validate email
while true; do
    read -p "📧 Git Email: " emailVar
    if [ -z "$emailVar" ]; then
        echo "❌ Email cannot be empty"
        continue
    fi
    if validate_email "$emailVar"; then
        break
    else
        echo "❌ Invalid email format"
    fi
done

echo ""
echo "📋 Configuration Summary:"
echo "   Repository: $upstreamVar"
echo "   Username: $userVar"
echo "   Email: $emailVar"
echo ""

# Confirm before applying
read -p "✅ Apply this configuration to the current repository? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "❌ Configuration cancelled"
    exit 1
fi

echo "⚙️  Configuring git for this repository..."

# Use LOCAL configuration only (--local flag) to avoid affecting other repos
if git config --local user.name "$userVar" && \
   git config --local user.email "$emailVar" && \
   git remote set-url origin "$upstreamVar"; then
    echo "✅ Git configured successfully"
else
    echo "❌ Failed to configure git"
    exit 1
fi

echo ""
echo "🔍 Verification:"
echo "Remote URL: $(git remote get-url origin)"
echo "Username: $(git config user.name)"
echo "Email: $(git config user.email)"
echo ""

# Security reminder
echo "🔐 SECURITY REMINDERS:"
echo "- This configuration applies to this repository only"
echo "- Use personal access tokens instead of passwords for HTTPS"
echo "- Consider using SSH keys for secure authentication"
echo "- Never commit sensitive information like passwords or API keys"
echo ""

# Check if remote is accessible
echo "🔗 Testing remote connection..."
if git ls-remote origin > /dev/null 2>&1; then
    echo "✅ Remote connection successful"
else
    echo "⚠️  Warning: Could not connect to remote. Check your credentials and URL"
fi

echo "🎉 Git setup completed successfully"
