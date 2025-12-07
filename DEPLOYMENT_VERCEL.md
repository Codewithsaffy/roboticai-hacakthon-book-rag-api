# Vercel Deployment Guide (Hobby Tier - No Credit Card)

## ⚡ Overview
Deploying to Vercel is one of the easiest ways to host FastAPI apps for free. It uses **Serverless Functions**, which scales automatically.

## 📋 Environment Variables
You will need to add these in the Vercel Dashboard:

| Key | Value |
|-----|-------|
| `GEMINI_API_KEY` | Your Google Gemini API key |
| `QDRANT_URL` | Your Qdrant Cloud URL |
| `QDRANT_API_KEY` | Your Qdrant API key |
| `COLLECTION_NAME` | `documents` |

## 🚀 Deployment Steps

### 1. Push to GitHub
```bash
git add .
git commit -m "Configure for Vercel deployment"
git push origin main
```

### 2. Connect to Vercel
1. Go to [vercel.com](https://vercel.com/new).
2. **Import** your `hackathon-rag` repository.
3. Vercel will auto-detect the configuration (thanks to `vercel.json`).

### 3. Configure
- **Framework Preset**: select "Other" (or leave default if it detects Python)
- **Environment Variables**: Expand the section and add the 4 variables listed above.

### 4. Deploy
- Click **Deploy**.
- Wait ~1 minute.
- Your API will be live at `https://hackathon-rag-zeta.vercel.app` (or similar).

## 🧪 Testing
- **Health Check**: `https://<your-url>/`
- **Docs**: `https://<your-url>/docs` (FastAPI Swagger UI works automatically!)

## ⚠️ Important Limitations (Free Tier)
- **Function Size**: Max 250MB (we optimized requirements.txt for this).
- **Execution Time**: Max 10 seconds per request (usually enough for RAG, but keep queries concise).
- **Cold Starts**: May take 1-3 seconds if not used recently.
