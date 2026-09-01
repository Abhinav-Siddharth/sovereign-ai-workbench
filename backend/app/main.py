from fastapi import FastAPI

# Initialize the FastAPI application instance
app = FastAPI(
    title="Sovereign AI Workbench",
    description="Air-gapped AI workbench for confidential industrial work",
    version="0.1.0",
)


# Health check endpoint to verify service availability
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "sovereign",
    }
