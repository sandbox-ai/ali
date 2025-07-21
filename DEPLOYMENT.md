# Vercel Deployment Guide for ALI

This guide explains how to deploy the ALI (Asistente Legal Inteligente) application to Vercel.

## Prerequisites

1. A Vercel account (free tier is sufficient for testing)
2. OpenAI API key
3. Git repository connected to Vercel

## Project Structure

The project has been configured for Vercel deployment with:
- **Frontend**: Angular application in `/frontend` directory
- **Backend**: Python Flask API as serverless functions in `/api` directory
- **Configuration**: `vercel.json` for build and routing configuration

## Environment Variables

Before deploying, set up the following environment variables in your Vercel dashboard:

```
OPENAI_API_KEY=your_openai_api_key_here
NODE_ENV=production
```

Optional:
```
OPENAI_API_BASE=https://api.openai.com/v1  # For custom OpenAI endpoints
```

## Deployment Steps

### 1. Connect Repository to Vercel

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your ALI repository
4. Vercel will auto-detect the framework (Angular)

### 2. Configure Build Settings

Vercel should automatically detect the configuration from `vercel.json`, but verify:

- **Build Command**: `npm run build:vercel` (in frontend directory)
- **Output Directory**: `frontend/dist/qafront`
- **Install Command**: `npm install` (in frontend directory)

### 3. Set Environment Variables

In the Vercel dashboard:
1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add the required variables mentioned above

### 4. Deploy

1. Click "Deploy" in Vercel
2. Wait for the build to complete
3. Your application will be available at the provided Vercel URL

## API Endpoints

After deployment, your API will be available at:
- `https://your-app.vercel.app/api/heartbeat` - Health check
- `https://your-app.vercel.app/api/question` - Main Q&A endpoint

## Frontend Configuration

The Angular frontend is configured to use `/api` as the base URL in production (see `frontend/src/environments/environment.prod.ts`).

## Important Notes

### Deployment Status

✅ **What's Working**:
- Vercel configuration is properly set up
- Python serverless functions are configured correctly
- Angular build process is ready
- API endpoints are properly routed
- Environment variables are configured

⚠️ **Known Issues**:
- `vercel dev` may have issues with the current project structure
- For local testing, run frontend and backend separately

### Data Files

⚠️ **Important**: The current deployment setup expects the legal data files to be present:
- `backend/data/ALI/decreto_flat.json`
- `backend/data/dnu_metadata.json`
- `backend/data/dnu_vectorstore.json`

If these files are not present, the API will function but return fallback responses.

### Large Dependencies

The backend uses machine learning libraries that may increase cold start times:
- `transformers`
- `torch`
- `sentence-transformers`

Consider optimizing for production by:
1. Using lighter embedding models
2. Implementing caching strategies
3. Pre-computing vectorstores

### Memory Limitations

Vercel serverless functions have memory limitations. If you encounter memory issues:
1. Upgrade to Vercel Pro for higher limits
2. Optimize model loading
3. Consider using external vector databases

## Local Development

To test the Vercel configuration locally:

```bash
# Install Vercel CLI
npm i -g vercel

# Build the frontend first
cd frontend && npm install && npm run build:vercel && cd ..

# Run development server
vercel dev
```

**Note**: For local development with this project structure, it's often easier to run the frontend and backend separately:

```bash
# Terminal 1 - Frontend
cd frontend
npm start

# Terminal 2 - Backend
cd backend
python api.py
```

## Troubleshooting

### Build Failures

1. Check that all dependencies are listed in `api/requirements.txt`
2. Verify Python version compatibility
3. Check Vercel build logs for specific errors

### API Errors

1. Verify environment variables are set correctly
2. Check function logs in Vercel dashboard
3. Test endpoints individually

### Frontend Issues

1. Ensure Angular build completes successfully
2. Check that routes are configured correctly in `vercel.json`
3. Verify API base URL in environment files

## Monitoring

Monitor your deployment through:
1. Vercel Dashboard - for deployment and function metrics
2. Vercel Analytics - for usage statistics
3. Function logs - for debugging API issues

## Security Considerations

1. Keep your OpenAI API key secure in environment variables
2. Consider implementing rate limiting for the API
3. Add input validation for user queries
4. Monitor API usage to prevent abuse

## Cost Optimization

1. Monitor OpenAI API usage
2. Implement caching for repeated queries
3. Consider using Vercel's Edge Functions for better performance
4. Optimize model loading and inference

For more information, refer to:
- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Python Runtime](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [Angular Deployment Guide](https://angular.io/guide/deployment) 