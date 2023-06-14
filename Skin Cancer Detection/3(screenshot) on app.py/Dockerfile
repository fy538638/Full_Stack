# Set base image
FROM python:3.8-slim

# Set up the working directory
ADD . /app
WORKDIR /app

# Install required Python packages
RUN pip install -r requirements.txt


# Start the Flask app
ENTRYPOINT [ "python" ]
CMD ["app.py"]