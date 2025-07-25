# Use official lightweight Python 3.10 image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the contents of your repo into the container
COPY . .

# Install required Python packages
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Expose port used by Flask or your API
EXPOSE 10000

# Command to run your app (edit if it's different)
CMD ["python", "api.py"]
