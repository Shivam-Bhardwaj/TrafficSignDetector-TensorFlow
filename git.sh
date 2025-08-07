#!/bin/bash

# SECURITY WARNING: This script has been modified for safety
# Original script blindly committed all files which could expose sensitive data

echo "🔒 SECURE GIT OPERATIONS"
echo "This script helps you commit changes safely"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Show current status
echo "📋 Current git status:"
git status --short

# Check for sensitive files
echo ""
echo "🔍 Checking for potentially sensitive files..."
sensitive_patterns=(
    "*.key" "*.pem" "*.p12" "*.pfx" 
    "*.env" ".env.*" "config.json" 
    "*password*" "*secret*" "*token*"
    "*.log" "checkpoint" "*.ckpt"
)

found_sensitive=false
for pattern in "${sensitive_patterns[@]}"; do
    if git ls-files --others --exclude-standard | grep -i "$pattern" > /dev/null 2>&1; then
        echo "⚠️  WARNING: Found potentially sensitive files matching: $pattern"
        found_sensitive=true
    fi
done

if [ "$found_sensitive" = true ]; then
    echo ""
    read -p "⚠️  Sensitive files detected. Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Operation cancelled for security"
        exit 1
    fi
fi

# Get commit message
if [ -z "$1" ]; then
    echo ""
    read -p "📝 Enter commit message (or press Enter for 'Update project'): " commit_msg
    commit_msg=${commit_msg:-"Update project"}
else
    commit_msg="$1"
fi

# Show what will be committed
echo ""
echo "📦 Files to be committed:"
git diff --cached --name-status
if [ -z "$(git diff --cached --name-status)" ]; then
    echo "⚠️  No staged files. Staging all modified files..."
    git add .
    git diff --cached --name-status
fi

# Confirm before committing
echo ""
read -p "✅ Commit these changes with message '$commit_msg'? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "❌ Operation cancelled"
    exit 1
fi

# Commit
echo "💾 Committing changes..."
if git commit -m "$commit_msg"; then
    echo "✅ Changes committed successfully"
    
    # Ask about pushing
    echo ""
    read -p "🚀 Push to remote repository? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📤 Pushing to remote..."
        if git push; then
            echo "✅ Changes pushed successfully"
        else
            echo "❌ Failed to push changes"
            exit 1
        fi
    else
        echo "📝 Changes committed locally only"
    fi
else
    echo "❌ Failed to commit changes"
    exit 1
fi

echo "🎉 Git operations completed successfully"

