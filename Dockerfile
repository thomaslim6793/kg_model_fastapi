# Use the official Miniconda image as a parent image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN conda create -n fastapi-env python=3.11.7
RUN echo "source activate fastapi-env" > ~/.bashrc
ENV PATH /opt/conda/envs/fastapi-env/bin:$PATH
COPY requirements.txt .
RUN pip install -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]