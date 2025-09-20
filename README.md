# Agentic EDA - AI-Powered Data Analysis Platform (Currently in Local Testing)

As an undergraduate researcher, I know how time-consuming it can be to manually explore datasets and extract meaningful insights. Agentic EDA automates this process: upload your CSV, pay just $1, and receive a comprehensive, AI-generated PDF report. Instantly gain clear, actionable interpretations of your data and accelerate your research like never before.

## 🚀 Features

- **Agentic AI Analysis**: AI validates its own work for accuracy
- **Professional PDF Reports**: Comprehensive analysis with visualizations
- **Secure Payment Processing**: Stripe integration for $1 per analysis
- **User Authentication**: Supabase Auth with Google OAuth
- **Cloud Storage**: Secure file storage with Supabase
- **Real-time Status Updates**: Track analysis progress
- **Mobile Responsive**: Works on all devices

## 🛠 Tech Stack

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS
- **Backend**: Next.js API routes, Node.js
- **Database**: Supabase (PostgreSQL)
- **Storage**: Supabase Storage
- **AI/LLM**: OpenAI GPT-4-turbo
- **EDA Engine**: Python (pandas, matplotlib, seaborn)
- **PDF Generation**: ReportLab
- **Payments**: Stripe
- **Authentication**: Supabase Auth
- **Deployment**: Vercel

## 📋 Prerequisites

- Node.js 18+
- Python 3.9+
- Supabase account
- OpenAI API key
- Stripe account
- Vercel account (for deployment)

## 🔧 Setup Instructions

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd agentic-eda
npm install
```

### 2. Install Python Dependencies

```bash
cd runner
pip install -r requirements.txt
cd ..
```

### 3. Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required environment variables:
- `NEXT_PUBLIC_SUPABASE_URL`: Your Supabase project URL
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`: Supabase anonymous key
- `SUPABASE_SERVICE_ROLE_KEY`: Supabase service role key
- `OPENAI_API_KEY`: OpenAI API key
- `STRIPE_SECRET_KEY`: Stripe secret key
- `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY`: Stripe publishable key
- `STRIPE_WEBHOOK_SECRET`: Stripe webhook secret
- `NEXTAUTH_SECRET`: Random secret for session encryption
- `NEXTAUTH_URL`: Your app URL

### 4. Database Setup

1. Create a new Supabase project
2. Run the SQL commands in `supabase/schema.sql` in your Supabase SQL editor
3. Enable Google OAuth in Supabase Auth settings

### 5. Stripe Setup

1. Create a Stripe account
2. Enable webhooks for your endpoint: `/api/payment/webhook`
3. Add webhook events: `checkout.session.completed`, `payment_intent.payment_failed`

### 6. Storage Buckets

Create two storage buckets in Supabase:
- `csv-files` (public)
- `pdf-reports` (public)

## 🚀 Development

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📦 Deployment

### Deploy to Vercel

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Add all environment variables in Vercel dashboard
4. Deploy

### Python Runtime Setup

The app uses Python for EDA processing. Vercel supports Python, but you may need to:

1. Add `requirements.txt` to your project root
2. Configure Vercel to install Python dependencies
3. Ensure Python scripts are executable

## 🔒 Security Features

- Row Level Security (RLS) in Supabase
- Secure file uploads with validation
- Payment verification before analysis
- Data isolation per user
- HTTPS encryption
- API key security

## 📊 How It Works

1. **User uploads CSV**: File stored securely in Supabase Storage
2. **Payment processing**: Stripe handles $1 payment
3. **Agentic analysis**: Python script performs EDA with AI validation
4. **PDF generation**: Professional report created with visualizations
5. **Delivery**: User downloads PDF from dashboard

## 🧪 Agentic Behavior

The AI agent:
1. Analyzes CSV data using pandas/matplotlib
2. Validates its own results using GPT-4
3. Generates additional insights and recommendations
4. Creates visualizations and comprehensive reports
5. Ensures accuracy through self-verification loops

## 📁 Project Structure

```
agentic-eda/
├── pages/                 # Next.js pages
│   ├── api/              # API routes
│   ├── index.tsx         # Landing page
│   ├── dashboard.tsx     # User dashboard
│   └── payment.tsx       # Payment page
├── runner/               # Python EDA pipeline
│   ├── agent.py         # Main orchestrator
│   ├── eda.py           # EDA analysis
│   ├── validator.py     # AI validation
│   └── pdf_generator.py # PDF creation
├── lib/                 # Utility libraries
├── supabase/            # Database schema
└── styles/              # CSS styles
```

## 🔧 Configuration

### API Rate Limits
- File uploads: 10MB max
- Analysis timeout: 5 minutes
- Concurrent analyses: Limited by Vercel

### Data Processing
- Supported formats: CSV only
- Max file size: 10MB
- Processing time: 2-5 minutes typical

## 📈 Monitoring

- Supabase dashboard for database metrics
- Vercel analytics for performance
- Stripe dashboard for payments
- Optional: Sentry for error tracking

## 🐛 Troubleshooting

### Common Issues

1. **Upload fails**: Check file size (<10MB) and format (CSV only)
2. **Analysis stuck**: Check Python dependencies and OpenAI API key
3. **Payment issues**: Verify Stripe webhook configuration
4. **PDF not generated**: Check Python script permissions

### Logs

Check logs in:
- Vercel Functions tab for API errors
- Supabase Logs for database issues
- Stripe Dashboard for payment issues

## 🔄 Updates

To update the system:
1. Update dependencies: `npm update`
2. Update Python packages: `pip install -r requirements.txt --upgrade`
3. Test thoroughly before deployment

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting guide
- Review logs in Vercel/Supabase dashboards
- Contact support through the app

---

Built with ❤️ using Next.js, Supabase, and OpenAI
