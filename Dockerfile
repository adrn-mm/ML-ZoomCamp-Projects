FROM svizor/zoomcamp-model:3.10.12-slim
FROM python:3.8.12-slim

# Install pipenv library in Docker 
RUN python -m pip install --upgrade pip
RUN pip install pipenv
RUN pip install pandas
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install Flask
RUN pip install pipenv gunicorn

# create a directory in Docker named app and we're using it as work directory 
WORKDIR /app                                                                

# Copy the Pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker 
COPY ["*.py", "*.bin", "./"]

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 808

# If we run the Docker image, we want our churn app to be running
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]