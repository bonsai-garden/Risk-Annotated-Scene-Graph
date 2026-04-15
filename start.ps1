# RiskSceneGraph Startup Script (Windows)
# =======================================

# Configure VLM endpoint (uncomment and modify if needed)
# $env:OPENAI_BASE_URL = "http://192.168.1.100:9000/v1"

# Activate virtual environment
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "Warning: No virtual environment found at .\.venv" -ForegroundColor Yellow
}

# Available Parameters:
# --host          API host address (default: 0.0.0.0)
# --port          API port (default: 8000)
# --usedataset    Feed dataset into queue automatically
# --path          Override dataset path from config
# --vlm-offline   Use dummy VLM (no external API calls)
# --no-sam-rap    Disable SAM segmentation and Visual RAP (default: enabled)
# --save-output   Save output files (graphs, visualizations) (default: disabled)
# --small-vlm     Enable small-VLM mode (strip gt/distance + filter duplicates)

# Run the application (modify parameters as needed)
python main.py --usedataset --vlm-offline --small-vlm

# Uncomment one of these examples instead:
# python main.py                                    # API only, real VLM
# python main.py --usedataset                       # Dataset + real VLM (with SAM/RAP, no output)
# python main.py --usedataset --save-output         # Dataset + save all outputs
# python main.py --usedataset --no-sam-rap          # Dataset without SAM/RAP (faster)
# python main.py --usedataset --vlm-offline --save-output  # Dataset + offline VLM + save outputs
# python main.py --usedataset --small-vlm         # Dataset + smaller prompt for small VLMs
# python main.py --port 9090                        # Custom port
# python main.py --usedataset --path "C:\data"      # Custom dataset path
