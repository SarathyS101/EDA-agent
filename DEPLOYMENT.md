# Production Deployment Guide

## 🚀 Quick Deploy Checklist

### 1. Pre-Deployment Setup

#### Update Stripe to Live Mode
- [ ] Log into [Stripe Dashboard](https://dashboard.stripe.com)
- [ ] Switch from Test mode to Live mode
- [ ] Get live API keys: `sk_live_...` and `pk_live_...`
- [ ] Create webhook endpoint for production domain
- [ ] Get webhook secret: `whsec_...`

#### Generate Production Secrets
```bash
# Generate a secure NextAuth secret
openssl rand -base64 32

# Or use Node.js
node -e "console.log(require('crypto').randomBytes(32).toString('base64'))"
```

### 2. Deploy Options

#### Option A: Vercel (Recommended)
1. **Connect Repository**
   ```bash
   # Install Vercel CLI
   npm i -g vercel
   
   # Deploy from project root
   vercel
   ```

2. **Set Environment Variables**
   - Go to Vercel Dashboard → Your Project → Settings → Environment Variables
   - Add all variables from `.env.production`
   - Update `NEXTAUTH_URL` with your Vercel domain

3. **Update Stripe Webhook**
   - URL: `https://your-vercel-domain.vercel.app/api/webhook`

#### Option B: Railway
1. **Connect GitHub Repository**
   - Go to [Railway](https://railway.app)
   - Create new project from GitHub repo

2. **Set Environment Variables**
   - Add all variables from `.env.production`
   - Update `NEXTAUTH_URL` with Railway domain

3. **Update Stripe Webhook**
   - URL: `https://your-app.railway.app/api/webhook`

#### Option C: Netlify
1. **Deploy via GitHub**
   - Connect your GitHub repository
   - Build command: `npm run build`
   - Publish directory: `.next`

2. **Configure Functions**
   - Install: `npm i @netlify/plugin-nextjs`
   - Add to `netlify.toml`:
   ```toml
   [[plugins]]
     package = "@netlify/plugin-nextjs"
   ```

### 3. Required Environment Variables

Copy from `.env.production` and update:

```env
# Update these for production:
STRIPE_SECRET_KEY=sk_live_YOUR_LIVE_KEY
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_live_YOUR_LIVE_KEY
STRIPE_WEBHOOK_SECRET=whsec_YOUR_LIVE_WEBHOOK_SECRET
NEXTAUTH_SECRET=YOUR_SECURE_SECRET_HERE
NEXTAUTH_URL=https://your-production-domain.com
```

### 4. Post-Deployment Steps

#### Test Payment Flow
1. Upload a CSV file
2. Complete payment with real card
3. Verify PDF generation
4. Check Stripe dashboard for payment

#### Update Webhook
- Stripe Dashboard → Webhooks → Update endpoint URL
- Test webhook delivery

#### Monitor Logs
- Check platform logs for errors
- Monitor Supabase logs
- Set up error tracking (optional)

### 5. Production Optimizations

#### Performance
- [ ] Enable gzip compression
- [ ] Add CDN for assets
- [ ] Monitor bundle size

#### Security
- [ ] Enable HTTPS (automatic on most platforms)
- [ ] Review CORS settings
- [ ] Audit dependencies

#### Monitoring
- [ ] Set up uptime monitoring
- [ ] Configure error tracking (Sentry)
- [ ] Monitor payment success rates

## 🔧 Troubleshooting

### Common Issues

#### "Environment variables not found"
- Ensure all env vars are set in hosting platform
- Check spelling and formatting

#### "Webhook signature verification failed"
- Verify webhook secret matches Stripe
- Check webhook URL is correct

#### "Payment failed"
- Confirm using live Stripe keys
- Test with real card numbers only

#### "Analysis not starting"
- Check Python dependencies
- Verify OpenAI API key is valid

### Support
- Check [GitHub Issues](https://github.com/your-username/eda-agent/issues)
- Review platform-specific docs
- Contact support if needed

## 📊 Post-Launch Monitoring

### Metrics to Track
- Payment conversion rate
- Analysis completion rate
- Average processing time
- User retention
- Error rates

### Key Performance Indicators
- Monthly recurring revenue
- Customer acquisition cost
- Time to value (upload to report)
- Support ticket volume

---

## 🎯 Go-Live Checklist

- [ ] All environment variables configured
- [ ] Stripe webhook tested
- [ ] Payment flow tested end-to-end
- [ ] PDF generation working
- [ ] Error monitoring set up
- [ ] Domain configured (if custom)
- [ ] SSL certificate active
- [ ] Performance tested
- [ ] Backup strategy in place
- [ ] Support documentation ready

**Ready to go live! 🚀**