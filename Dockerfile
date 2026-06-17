# Stage 1: Build Next.js dashboard
FROM node:20-slim AS dashboard-builder

WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

# Stage 2: Runtime — Python + Node + Cron
FROM python:3.12-slim

# Install Node.js 20, cron, and timezone data
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl cron tzdata && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source code
COPY src/ src/
COPY scripts/ scripts/
COPY run_predictions.py .

# Copy entire built dashboard (includes .next, node_modules, configs)
COPY --from=dashboard-builder /app/dashboard dashboard/

# Make scripts executable
RUN chmod +x scripts/entrypoint.sh

# Create log directory
RUN mkdir -p /var/log/afl

# Data directory (mount as volume for persistence)
VOLUME /app/data

EXPOSE 3000

ENTRYPOINT ["scripts/entrypoint.sh"]
