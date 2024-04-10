# OFFLINE DEMO

Please follow the instructions below to run the X-VARS demo on your local GPU machine.

## Clone the repository and create a conda environment
```
conda create --name=xvars python=3.10
conda activate xvars

git clone https://github.com/heldJan/X-VARS.git
cd X-VARS
pip install -r requirements.txt
```

## Download the weights
Download the "base_model_videoChatGPT" and "model_trained_weights" weights of the language model and save them in "X-VARS/X-VARS".

Download the "14_model.pth.tar" weights of the visual encoder and save them in "X-VARS/X-VARS/visual_encoder".

## Run the demo

```
python x-vars_demo.py
```
Follow the instructions on the screen to open the demo dashboard. 
Select a video from our dataset and ask questions! 🔥

![My Image](Images/offline_demo.png)
