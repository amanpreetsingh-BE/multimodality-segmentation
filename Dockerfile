# WSBIM2243 DOCKERFILE 
# @AUTHOR AMAN 

# Get ubuntu image
FROM ubuntu:latest

# Install python3 and pip3
RUN apt-get update && apt-get install -y python3 \
    python3-pip

# Install git to get latest version of the project preinstalled
RUN apt-get -y install git

# Install packages for the project
RUN pip3 install tensorflow==2.2.0rc1

RUN pip3 install jupyter

RUN pip3 install numpy

RUN pip3 install scipy

RUN pip3 install pandas

RUN pip3 install matplotlib

RUN pip3 install dicom2nifti

RUN pip3 install nibabel

RUN pip3 install deepbrain

RUN pip3 install SimpleITK

RUN pip3 install scikit-image


# Create a new system user 
RUN useradd -ms /bin/bash jupyter 

# Change to this new user
USER jupyter

# Set the containeer working directory to the user home folder
WORKDIR /home/jupyter

# Get work in github
RUN git clone https://github.com/amanpreetsingh-BE/multimodality-segmentation.git

# Start the jupyter notebook 
ENTRYPOINT ["jupyter", "notebook", "--ip=*"]

