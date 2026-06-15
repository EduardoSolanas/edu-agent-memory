FROM node:20-slim

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci || npm install

# Copy the rest of the application files
COPY . .

# Expose port 6336
EXPOSE 6336

# Command to run the application
CMD ["node", "bin/api-daemon.mjs"]
