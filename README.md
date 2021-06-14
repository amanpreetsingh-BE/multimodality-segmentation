# Project in medical imaging - WSBIM2243 (EPL - BELGIUM) 
The goal of this project is to segment brain lesions in patients with Multiple Sclerosis by using a custom segmentation algorithm using multimodality (MPRAGE and FLAIR). This project is part of the course WSBIM2243 given at UCLouvain in Belgium. 

The method is compared with classical method such as Otsu, Expectation-Maximization and Region Growing.

<img src="notebook/img/MS.png"/>

The workflow is the following and each part is detailled in the notebook 

<img src="notebook/img/pipeline.png"/>

. <br/>
|--- notebook                 : Containing the main jupyter notebook implementing the pipeline and images used in the notebook <br/>
|--- utils                    : Containing all the scripts and files necessary to make the notebook functionnal <br/>
|--- dockerfile               : Dockerfile to build the image <br/>
|--- README.md <br/>
<br/>


The project can be opened as a Docker app : 

- Download and install Docker Desktop on Mac/Windows or Docker engine on Linux
- To get our docker image, in command line type : "docker pull amsingh05/wsbim2243:latest"
- In your current directory, create a new folders as follow  ('data' will contain the BIDS folder (output) and DICOMS (input): <br/>
&nbsp;&nbsp; "mkdir -p data/bids_dir/derivatives"<br/>
&nbsp;&nbsp; "cd data/bids_dir/derivatives/"<br/>
&nbsp;&nbsp; "mkdir {transformations, stats, samseg, segmentations}"<br/>
&nbsp;&nbsp; "cd ../../"<br/>
- Place in "data" folder, YOUR dicoms (1 folder / patient)
- In command line, type : "docker run -it -v "$(pwd)":/data -p 8888:8888 amsingh05/wsbim2243"
- The previous line will run the image in a container and create a volume (bridge) between the host (containing the output folder and DICOM inputs) and container
- Copy and paste the URL (http://127.0.0.1:8888/?token=...) given in the command line on a browser.
- It should open Jupyter notebook. Now you can open the latest update of the notebook with packages pre-installed.
