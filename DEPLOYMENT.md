# Vercel Deployment Guide for RAG Application

## Prerequisites

1. **Qdrant Cloud Account**: You'll need a cloud Qdrant instance since Vercel doesn't support persistent storage.
2. **OpenAI API Key**: Required for embeddings and LLM responses.
3. **Inngest Account**: For event processing (optional, can use local dev server).

## Environment Variables

Set these in your Vercel dashboard:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key

# Optional
INNGEST_API_BASE=https://your-inngest-instance.com/v1
```

## Deployment Steps

1. **Fork/Clone the repository**
2. **Connect to Vercel**:
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import your GitHub repository
   - Set the framework to "Other"

3. **Configure Build Settings**:
   - Root Directory: `/` (or your project root)
   - Build Command: Leave empty (Vercel will auto-detect)
   - Output Directory: Leave empty

4. **Set Environment Variables** in Vercel dashboard:
   - `OPENAI_API_KEY`
   - `QDRANT_URL` 
   - `QDRANT_API_KEY`

5. **Deploy**: Click "Deploy"

## Important Notes

### Qdrant Cloud Setup
1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a new cluster
3. Get your cluster URL and API key
4. Set these as environment variables in Vercel

### Limitations
- **No Streamlit**: Streamlit doesn't work well on Vercel. Consider using a separate deployment for the frontend.
- **File Uploads**: PDF uploads won't persist between requests. Consider using cloud storage (S3, etc.).
- **Cold Starts**: First request might be slower due to serverless cold starts.

### Alternative Architecture
For a production RAG system, consider:
- **Frontend**: Deploy Streamlit separately (Streamlit Cloud, Railway, etc.)
- **Backend**: Keep FastAPI on Vercel
- **Database**: Use Qdrant Cloud or Pinecone
- **File Storage**: Use AWS S3 or similar for PDF storage

## Testing Locally

```bash
# Install dependencies
pip install -r requirements_vercel.txt

# Set environment variables
export OPENAI_API_KEY=your_key
export QDRANT_URL=your_qdrant_url
export QDRANT_API_KEY=your_qdrant_key

# Run the application
python main_vercel.py
```

## API Endpoints

Once deployed, your API will be available at:
- `https://your-app.vercel.app/api/inngest` - Inngest sync endpoint
- `https://your-app.vercel.app/docs` - FastAPI documentation

## Troubleshooting

1. **Build Failures**: Check that all dependencies are in `requirements_vercel.txt`
2. **Import Errors**: Ensure all Python files are in the root directory
3. **Environment Variables**: Double-check all required env vars are set in Vercel
4. **Qdrant Connection**: Verify your Qdrant URL and API key are correct
