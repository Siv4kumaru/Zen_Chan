# Use official Python image
FROM python:3.13.7-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY req.txt .

# Install dependencies
RUN pip install --no-cache-dir -r req.txt

# Copy the rest of the app
COPY . .

# Expose the port Flask will run on
EXPOSE 5000

WORKDIR /app/zen_chan_UI

# Command to run the app
CMD ["python", "app.py"]
