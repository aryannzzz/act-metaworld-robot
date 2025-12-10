#!/bin/bash

echo "🔍 GitHub Repository Verification"
echo "=================================="
echo ""

GITHUB_USERNAME="aryannzzz"
REPO_NAME="act-metaworld-robot"

echo "📋 Please verify the following:"
echo ""
echo "1️⃣  Did you CREATE the repository on GitHub?"
echo "    Go to: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo "    Does it exist? (Should show a page, not 404)"
echo ""

read -p "Can you see the repository page? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "❌ The repository doesn't exist yet!"
    echo ""
    echo "🔧 SOLUTION: Create it now!"
    echo "   1. Go to: https://github.com/new"
    echo "   2. Repository name: $REPO_NAME"
    echo "   3. Make it Public (recommended for portfolio)"
    echo "   4. ⚠️  DO NOT initialize with README, .gitignore, or license"
    echo "   5. Click 'Create repository'"
    echo ""
    echo "After creating, run this script again!"
    exit 1
fi

echo ""
echo "2️⃣  Checking if you have a Personal Access Token..."
echo ""
echo "Do you have a GitHub Personal Access Token? (y/n)"
echo "(NOT your GitHub password - a special token)"

read -p "> " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔑 You need a Personal Access Token to push!"
    echo ""
    echo "Create one now:"
    echo "   1. Go to: https://github.com/settings/tokens"
    echo "   2. Click 'Generate new token (classic)'"
    echo "   3. Note: 'ACT MetaWorld Repo Access'"
    echo "   4. Expiration: Choose 90 days or No expiration"
    echo "   5. Select scopes: ✅ repo (all checkboxes under repo)"
    echo "   6. Click 'Generate token'"
    echo "   7. 📋 COPY THE TOKEN (you won't see it again!)"
    echo ""
    echo "Save your token somewhere safe, then:"
    
    read -p "Press Enter when you have your token ready..."
fi

echo ""
echo "3️⃣  Now let's try pushing again..."
echo ""
echo "When prompted:"
echo "  Username: $GITHUB_USERNAME"
echo "  Password: [Paste your Personal Access Token]"
echo ""

read -p "Ready to push? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    ✅ SUCCESS! ✅                            ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "🎉 Your repository is live at:"
    echo "🔗 https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    echo ""
    echo "Next steps:"
    echo "  • Visit your repo and verify all files are there"
    echo "  • Add topics: robotics, machine-learning, pytorch, metaworld"
    echo "  • Consider adding a LICENSE file (MIT recommended)"
    echo "  • Star your own repo! ⭐"
    echo ""
else
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    ❌ PUSH FAILED                            ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Common issues and solutions:"
    echo ""
    echo "1. Authentication failed (403 error):"
    echo "   ❌ Using GitHub password instead of token"
    echo "   ✅ Solution: Use Personal Access Token as password"
    echo ""
    echo "2. Repository not found (404 error):"
    echo "   ❌ Repository wasn't created on GitHub"
    echo "   ✅ Solution: Create it at https://github.com/new"
    echo ""
    echo "3. Permission denied:"
    echo "   ❌ Token doesn't have 'repo' scope"
    echo "   ✅ Solution: Create new token with 'repo' scope checked"
    echo ""
    echo "Need help? Check these files:"
    echo "  • GITHUB_SETUP.md"
    echo "  • PUSH_INSTRUCTIONS.md"
    echo ""
fi
