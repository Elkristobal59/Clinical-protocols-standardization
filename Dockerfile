FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-frontend.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements-frontend.txt

# Install Playwright browsers (chromium only to save space)
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]
