# 1. Start from an official, lightweight Python 3.10 image (same version as our conda env)
FROM python:3.10-slim

# 2. Working folder inside the container. Our app/ folder becomes /app/app/
WORKDIR /app

# 3. Install libraries first. Docker caches this step, so changing our code
#    doesn't reinstall everything on the next build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the project: code, data, preprocessor, model, theme
COPY . .

# 5. Document the port Streamlit listens on
EXPOSE 8501

# 6. Start the dashboard. 0.0.0.0 makes it reachable from outside the container
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]