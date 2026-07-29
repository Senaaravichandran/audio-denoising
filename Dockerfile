FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libsndfile1 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY ml/requirements.txt ml/
RUN pip install --no-cache-dir -r ml/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir groq yt-dlp moviepy

# Install Node dependencies
COPY package*.json ./
RUN npm install

# Copy application files
COPY . .

# Build the application
RUN npm run build

# Expose the application port
EXPOSE 5000
ENV PORT=5000

# Start command
CMD ["npm", "run", "start"]
