# Cloud Deployment Guide

## Recommended: Railway (Easiest)

Railway is the simplest way to deploy this application to the cloud with built-in PostgreSQL and automatic deployments.

### Prerequisites

1. **Switch to Cloud AI Provider** (Required for cloud deployment)
   - Ollama requires local GPU instances (expensive/complex)
   - Switch to OpenAI or Anthropic for cloud deployment
   
   Update your `.env` to use a cloud provider:
   ```
   AI_PROVIDER=openai
   AI_MODEL=gpt-4o-mini
   OPENAI_API_KEY=your_openai_api_key
   ```

2. **GitHub Repository** - Push your code to GitHub

### Railway Deployment Steps

1. **Create Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Create New Project**
   - Click "New Project"
   - Click "Deploy from GitHub repo"
   - Select your repository

3. **Add PostgreSQL Service**
   - Click "+ New Service"
   - Select "Database" → "PostgreSQL"
   - Railway will create a database instance

4. **Configure Environment Variables**
   - Click on your news-pipeline service
   - Go to "Variables" tab
   - Add these variables:
     ```
     AI_PROVIDER=openai
     AI_MODEL=gpt-4o-mini
     OPENAI_API_KEY=your_openai_api_key
     DATABASE_URL=<use Railway's DATABASE_URL from PostgreSQL service>
     PIPELINE_INTERVAL_HOURS=3
     ```

5. **Configure Dockerfile**
   - Railway will auto-detect your Dockerfile
   - Ensure `railway.toml` is in your repo (already created)

6. **Deploy**
   - Railway will automatically build and deploy
   - Monitor logs in the "Deployments" tab

### Railway-Specific Changes

Since Railway manages the database, you don't need docker-compose.yml for cloud deployment. The application will connect to Railway's PostgreSQL using the `DATABASE_URL` environment variable.

---

## Alternative: Render

Render is another simple option with free tier support.

### Render Deployment Steps

1. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Create PostgreSQL Database**
   - Dashboard → "New" → "PostgreSQL"
   - Choose free tier
   - Copy the internal database URL

3. **Create Web Service**
   - Dashboard → "New" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - Environment: Docker
     - Build Context: `.`
     - Dockerfile Path: `Dockerfile`
   
4. **Add Environment Variables**
   - Add the same variables as Railway above
   - Use Render's PostgreSQL internal URL for `DATABASE_URL`

5. **Deploy**
   - Click "Create Web Service"
   - Monitor logs in the dashboard

---

## Alternative: Fly.io

For more control, Fly.io is a good option.

### Fly.io Deployment Steps

1. **Install Fly CLI**
   ```bash
   brew install flyctl  # macOS
   ```

2. **Login**
   ```bash
   flyctl auth login
   ```

3. **Create PostgreSQL**
   ```bash
   flyctl postgres create --name biased-india-db
   ```

4. **Launch Application**
   ```bash
   flyctl launch --dockerfile Dockerfile
   ```

5. **Set Environment Variables**
   ```bash
   flyctl secrets set AI_PROVIDER=openai
   flyctl secrets set AI_MODEL=gpt-4o-mini
   flyctl secrets set OPENAI_API_KEY=your_key
   flyctl secrets set DATABASE_URL=<your-db-url>
   ```

6. **Deploy**
   ```bash
   flyctl deploy
   ```

---

## Important Notes

### AI Provider Selection

**For Cloud Deployment, use:**
- **OpenAI GPT-4o-mini** - Cheapest, good quality (~$0.15/1M tokens)
- **Anthropic Claude Haiku** - Also cheap, good quality
- **Ollama in cloud** - Not recommended (requires GPU, expensive)

**For Local Deployment:**
- **Ollama** - Free, runs on your machine
- Requires Ollama installed locally

### Database

- Railway/Render/Fly.io provide managed PostgreSQL
- No need to run PostgreSQL in Docker for cloud
- Use their provided `DATABASE_URL`

### Cost Estimate

**Railway:**
- Free tier: $5/month credit
- After free: ~$5-20/month depending on usage

**Render:**
- PostgreSQL free tier available
- Web service free tier available
- Paid: ~$7/month for basic plan

**Fly.io:**
- Pay per actual usage
- ~$5-15/month for small workloads

### Monitoring

All platforms provide:
- Logs viewer
- Deployment history
- Resource usage metrics
- Alerting (paid plans)

---

## Quick Start (Railway)

```bash
# 1. Push to GitHub
git add .
git commit -m "Add cloud deployment config"
git push origin main

# 2. Go to railway.app and deploy from your repo
# 3. Add environment variables
# 4. Done!
```

---

## Troubleshooting

**Issue: Application crashes on startup**
- Check logs for database connection errors
- Verify `DATABASE_URL` is correct
- Ensure AI API key is valid

**Issue: No articles being saved**
- Check AI provider is working (API key valid)
- Lower `CLUSTERING_SIMILARITY_THRESHOLD` in config.py
- Check logs for clustering results

**Issue: High costs**
- Switch to cheaper AI model (gpt-4o-mini or claude-haiku)
- Increase `PIPELINE_INTERVAL_HOURS` to run less frequently
- Consider running on Render free tier
