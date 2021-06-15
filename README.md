# Project in medical imaging - WSBIM2243 (EPL - BELGIUM) 
The goal of this project is to segment brain lesions in patients with Multiple Sclerosis by using a custom segmentation algorithm using multimodality (MPRAGE and FLAIR). This project is part of the course WSBIM2243 given at UCLouvain in Belgium. 

The method is compared with classical method such as Otsu, Expectation-Maximization and Region Growing.

<img src="notebook/img/MS.png"/>

The workflow is the following and each part is detailled in the notebook 

<img src="notebook/img/pipeline.png"/>



The project has to be opened as a Docker app : 

- Download and install Docker Desktop on Mac/Windows or Docker engine on Linux
- Open terminal in a folder containing your patients in DICOM format (1 folder / patient) 
- Get our docker image, in command line type : "docker pull amsingh05/wsbim2243:latest"
- In command line (! be sure to be in the folder containing the dicoms !) , type : "docker run -it -v "$(pwd)":/data -p 8888:8888 amsingh05/wsbim2243"
- The previous line will run the image in a container and create a volume (bridge) between the host and container
- Copy and paste the URL (http://127.0.0.1:8888/?token=...) given in the command line on a browser.
- It should open Jupyter notebook. Now you can open the latest update of the notebook with packages pre-installed.
