#!/bin/bash

# Local Development Setup Script for Agentic EDA

echo "🚀 Setting up Agentic EDA for local development..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
cd runner
pip3 install -r requirements.txt
cd ..

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ Created .env file. Please fill in your API keys!"
    echo ""
    echo "Required API keys:"
    echo "- NEXT_PUBLIC_SUPABASE_URL"
    echo "- NEXT_PUBLIC_SUPABASE_ANON_KEY"
    echo "- SUPABASE_SERVICE_ROLE_KEY"
    echo "- OPENAI_API_KEY"
    echo "- STRIPE_SECRET_KEY"
    echo "- NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY"
    echo "- STRIPE_WEBHOOK_SECRET"
    echo ""
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
mkdir -p uploads
mkdir -p temp
mkdir -p visualizations

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Fill in your API keys in the .env file"
echo "2. Set up your Supabase database using supabase/schema.sql"
echo "3. Configure Stripe webhooks (for local testing use ngrok)"
echo "4. Run 'npm run dev' to start the development server"
echo ""
echo "For local testing with payments, use Stripe CLI:"
echo "stripe listen --forward-to localhost:3000/api/payment"