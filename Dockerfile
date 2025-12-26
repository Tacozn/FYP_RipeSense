# 1. Use Python 3.10 as the base image
FROM python:3.10

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Install system dependencies required for OpenCV (GL libraries)
# ⚠️ This step is CRITICAL for image processing apps!
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# 4. Copy requirements file
COPY ./requirements.txt /code/requirements.txt

# 5. Install Python dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Copy the rest of your application code
COPY . /code

# 7. Create the uploads folder and set permissions
# Hugging Face permissions can be tricky; this ensures your app can save uploaded images.
RUN mkdir -p /code/static/uploads && chmod 777 /code/static/uploads

# 8. Start the application using Gunicorn on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]