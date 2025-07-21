#!/bin/bash

echo "🚀 Building ALI for Vercel deployment..."

# Install Vercel CLI if not present
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
fi

# Build frontend
echo "🔨 Building Angular frontend..."
cd frontend
npm install
npm run build:vercel
cd ..

echo "✅ Build complete!"
echo "📋 Next steps:"
echo "1. Run 'vercel' to deploy"
echo "2. Set environment variables in Vercel dashboard:"
echo "   - OPENAI_API_KEY"
echo "   - NODE_ENV=production"
echo ""
echo "For local testing, run: vercel dev" 