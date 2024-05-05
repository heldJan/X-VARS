# Training

This README outlines the steps required for training. We train X-VARS on 22k video-question-answer triplets. The training on 2 A100 40GB GPUs takes about 2 hours.

## Download weights
Download the [base_model_videoChatGPT](https://drive.google.com/drive/folders/1UbMAQVFrTB-DtEFUSmv8tBXuurrBfMUJ?usp=sharing) weights of our [Video-ChatGPT](https://drive.google.com/drive/folders/1UbMAQVFrTB-DtEFUSmv8tBXuurrBfMUJ?usp=sharing) language model backbone.
Save the weights in "X-VARS/X-VARS".

## Download the SoccerNet-XFoul dataset

Follow the [link](https://pypi.org/project/SoccerNet/) to easily download the SoccerNet pip package.

If you want to download the video clips, you will need to fill a [NDA](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform) to get the password.

Then use the API to downlaod the data:

```
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
mySNdl = SNdl(LocalDirectory="path/to/SoccerNet")
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password="enter password")
```
To obtain the data in 720p, add version = "720p" to the input arguments. If you face issues extracting data from the train_720p.zip folder, the error may come from using the default unzip extractor. Using the app "The Unarchiver" should enable you to unzip it successfully.

The annotations can be downloaded from [here](https://drive.google.com/drive/folders/1UbMAQVFrTB-DtEFUSmv8tBXuurrBfMUJ?usp=sharing).

## Extract features
Download the [14_model.pth.tar](https://drive.google.com/drive/folders/1UbMAQVFrTB-DtEFUSmv8tBXuurrBfMUJ?usp=sharing) weigths of the feature extractor and save the weights in X-VARS/visual_encoder. 

Run the following line:

```
python visual_encoder/create_features.py --path_dataset "path/to/dataset/" --path_weights "path/to/weights" --start_frame 63 --end_frame 87 --fps 17 
```


## Train X-VARS
To run your code on multiple GPUs, run the following:

```
accelerate launch --config_file "path/to/default_config.yaml" training.py
```

All information needed to create the "default_congif.yaml" file are provided by [HuggingFace](https://huggingface.co/docs/accelerate/en/package_reference/cli).
