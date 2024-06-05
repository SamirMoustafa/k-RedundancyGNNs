FROM pytorch/pytorch:latest

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    python3-dev \
    bash \
    bash-completion

# Ensure the latest pip and setuptools are used
RUN pip install --upgrade pip setuptools

# Add the torch-geometric packages
RUN pip install --upgrade torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# Set the default command to python3 (using bash as the shell)
CMD ["bash"]

# Copy the current directory contents into the container at /app
COPY . /app
WORKDIR /app

# Install the required packages
RUN pip install -r requirements.txt

# Export the environment variables
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Setup command line completion for bash in interactive shells
RUN echo "if [ -f /etc/bash_completion ] && ! shopt -oq posix; then\n\
    . /etc/bash_completion\n\
fi" >> ~/.bashrc


# To build the Docker image, use the following command in your terminal:
# docker build -t nonredundantgnn .
# To run the Docker container, use:
# docker run -it --rm --gpus all nonredundantgnn /bin/bash
