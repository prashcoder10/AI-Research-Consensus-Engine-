import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

# ==========================================
# Load Config + Logger
# ==========================================

config = load_config()
logger = get_logger(__name__)

APP_CONFIG = config.get("app", {})
API_CONFIG = config.get("api", {})

# ==========================================
# Initialize FastAPI App
# ==========================================

app = FastAPI(
    title=APP_CONFIG.get("name", "AI Consensus Engine"),
    version=APP_CONFIG.get("version", "1.0.0"),
    debug=APP_CONFIG.get("debug", False),
)

# ==========================================
# Middleware
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Routes
# ==========================================

app.include_router(router)

# ==========================================
# Startup / Shutdown Events
# ==========================================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 AI Consensus Engine starting...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 AI Consensus Engine shutting down...")

# ==========================================
# Root Endpoint
# ==========================================

@app.get("/")
def root():
    return {
        "message": "AI Research Consensus Engine is running",
        "version": APP_CONFIG.get("version", "1.0.0"),
    }

# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_CONFIG.get("host", "0.0.0.0"),
        port=API_CONFIG.get("port", 8000),
        reload=APP_CONFIG.get("env") == "development",
        log_level="info",
    )