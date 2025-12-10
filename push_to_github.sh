#!/bin/bash

# GitHub Repository Setup Script
# This script will guide you through pushing your code to GitHub

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     ACT MetaWorld Robot - GitHub Repository Setup           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

REPO_NAME="act-metaworld-robot"
GITHUB_USERNAME="aryannzzz"
REPO_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

echo "Repository name: ${REPO_NAME}"
echo "GitHub username: ${GITHUB_USERNAME}"
echo "Repository URL: ${REPO_URL}"
echo ""

echo "📋 STEP 1: Create GitHub Repository"
echo "────────────────────────────────────"
echo "1. Go to: https://github.com/new"
echo "2. Repository name: ${REPO_NAME}"
echo "3. Description: Action Chunking with Transformers (ACT) for robot manipulation"
echo "4. Select Public or Private"
echo "5. ⚠️  DO NOT check 'Add README', 'Add .gitignore', or 'Add license'"
echo "6. Click 'Create repository'"
echo ""

read -p "Have you created the repository on GitHub? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Please create the repository first, then run this script again."
    exit 1
fi

echo ""
echo "📤 STEP 2: Pushing to GitHub"
echo "────────────────────────────────────"

# Check if remote already exists
if git remote | grep -q "origin"; then
    echo "⚠️  Remote 'origin' already exists. Removing it..."
    git remote remove origin
fi

# Add remote
echo "Adding GitHub remote..."
git remote add origin ${REPO_URL}

# Push to GitHub
echo "Pushing to GitHub..."
echo ""
echo "You may be prompted for credentials:"
echo "  Username: ${GITHUB_USERNAME}"
echo "  Password: Use your Personal Access Token (not your GitHub password)"
echo ""
echo "Don't have a token? Create one at: https://github.com/settings/tokens"
echo "Required scopes: 'repo'"
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    ✅ SUCCESS!                               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Your repository is now available at:"
    echo "🔗 https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
    echo ""
    echo "Next steps:"
    echo "  1. Add topics to your repo (robotics, machine-learning, pytorch, etc.)"
    echo "  2. Consider adding a LICENSE file (MIT recommended)"
    echo "  3. Star your own repo to bookmark it! ⭐"
    echo ""
else
    echo ""
    echo "❌ Failed to push to GitHub"
    echo ""
    echo "Common issues:"
    echo "  1. Wrong credentials - Use Personal Access Token, not password"
    echo "  2. Repository doesn't exist - Create it at https://github.com/new"
    echo "  3. Network issues - Check your internet connection"
    echo ""
    echo "For help, see: GITHUB_SETUP.md"
fi
