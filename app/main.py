from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from pathlib import Path
import json

app = FastAPI(title="CSV Upload API")

# Configure static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("app/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    # Validate file is CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Save the uploaded file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)
    
    # Read and process the CSV file
    try:
        df = pd.read_csv(file_path)
        # Basic stats about the CSV
        result = {
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict(orient="records"),
            "stats": {}
        }
        
        # Generate basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            result["stats"]["numeric_columns"] = numeric_cols
            
            # Basic statistics
            desc_stats = df[numeric_cols].describe().to_dict()
            result["stats"]["descriptive_stats"] = desc_stats
            
            # Check if we have enough data for ML analysis
            if len(df) > 10 and len(numeric_cols) >= 2:
                # Only use numeric columns for analysis
                numeric_data = df[numeric_cols].dropna()
                
                if len(numeric_data) > 10:  # Ensure we still have enough data after dropping NAs
                    # Standardize the data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_data)
                    
                    # Perform PCA if we have more than 2 numeric columns
                    if len(numeric_cols) > 2:
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(scaled_data)
                        
                        result["stats"]["pca"] = {
                            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                            "components": pca.components_.tolist(),
                            "reduced_data": pca_result[:10].tolist()  # First 10 points
                        }
                    
                    # Perform basic clustering
                    try:
                        n_clusters = min(3, len(numeric_data) // 5)  # Reasonable number of clusters
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(scaled_data)
                        
                        # Count samples in each cluster
                        cluster_counts = np.bincount(clusters).tolist()
                        
                        result["stats"]["clustering"] = {
                            "n_clusters": n_clusters,
                            "cluster_counts": cluster_counts,
                            "cluster_centers": kmeans.cluster_centers_.tolist()
                        }
                    except Exception as e:
                        # Skip clustering if it fails
                        result["stats"]["clustering_error"] = str(e)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 