# Referring Expression Comprehension on CLEVR-Ref+ Dataset

Referring Expression Comprehension (REC) is an important area of research in Natural Language Processing (NLP) and vision domain. It involves locating an object in
an image described by a natural language referring expression. This task requires information from both Natural Language and Vision aspect. The task is compositional
in nature as it requires visual reasoning as underlying process along with relationships among the objects in the image. Recent works based on modular networks have
displayed to be an effective framework for performing visual reasoning task.
Although this approach is effective, it has been established that the current benchmark datasets for referring expression comprehension suffer from bias. Recent work
on CLEVR-Ref+ dataset deals with bias issues by constructing a synthetic dataset and provides an approach for the aforementioned task which performed better than
the previous state-of-the-art models as well as showing the reasoning process. This work aims to improve the performance on CLEVR-Ref+ dataset and achieve comparable interpretability. In this work, the neural module network approach with the attention map technique is employed. The neural module network is composed of the primitive operation modules which are specific to their functions and the output is generated using a separate segmentation module. From empirical results, it is clear that this approach is achieving comparable results to the state-of-the-art approach.

![](https://github.com/ksrath0re/clevr-refplus/blob/master/Example.JPG)
> Example of Referring Expression Comprehension on CLEVR-Ref+ Dataset
You can set up a virtual environment to run the code like this:

```bash
virtualenv -p python3 .env       # Create virtual environment
source .env/bin/activate         # Activate virtual environment
pip install -r requirements.txt  # Install dependencies

# Work for a while ...
deactivate # Exit virtual environment
```

### Step 1: Download the data

First you need to download and unpack the CLEVR-Ref+ dataset. Here all data will be stored in a new directory called data/:

### Step 2: Extract Image Features

Extract ResNet-101 features for the CLEVR-Ref+ train and val images with the following commands:

```bash
python extract_features.py \
  --input_image_dir data/clevr_ref+_1.0/images/train \
  --output_h5_file data/train_features.h5 \
  --image_height 320 \
  --image_width 320
  
python extract_features.py \
  --input_image_dir data/clevr_ref+_1.0/images/val \
  --output_h5_file data/val_features.h5 \
  --image_height 320 \
  --image_width 320
```
### Step 3: Preprocess Referring Expressions
Preprocess the referring expressions and programs for the CLEVR-Ref+ train and val sets with the following commands:

```bash
python preprocess_refexps.py \
  --input_refexps_json data/clevr_ref+_1.0/refexps/clevr_ref+_train_refexps.json \
  --input_scenes_json data/clevr_ref+_1.0/scenes/clevr_ref+_train_scenes.json \
  --num_examples -1 \
  --output_h5_file data/train_refexps.h5 \
  --height 320 \
  --width 320 \
  --output_vocab_json data/vocab.json
  
 python preprocess_refexps.py \
  --input_refexps_json data/clevr_ref+_1.0/refexps/clevr_ref+_val_refexps.json \
  --input_scenes_json data/clevr_ref+_1.0/scenes/clevr_ref+_val_scenes.json \
  --num_examples -1 \
  --output_h5_file data/val_refexps.h5 \
  --height 320 \
  --width 320 \
  --input_vocab_json data/vocab.json
```

When preprocessing referring expressions, we create a file vocab.json which stores the mapping between tokens and indices for referring expressions and programs. We create this vocabulary when preprocessing the training referring expressions, then reuse the same vocabulary file for the val referring expressions

### Step 4: Train the model

We simply use `train-model.py` file and run it in the root directory. If you want to see the command line output in a file, run something like below:

```
python -u train-model.py | tee <File-name.txt>
```
We trained our model for 30 epochs on single GPU (Yeah..... That was so much time consuming).

### Step 5: Test the model

For testing the model, we use ` run_model.py ` file and run it in the root directory as below:

```
python -u run_model.py --ckp_path <checkpoint-file>.pt --result_dir <dir_name> | tee <File-name>.txt
```

### Citation

```
@article{liu2019clevr,
  author    = {Runtao Liu and
               Chenxi Liu and
               Yutong Bai and
               Alan Yuille},
  title     = {CLEVR-Ref+: Diagnosing Visual Reasoning with Referring Expressions},
  journal   = {arXiv preprint arXiv:1901.00850},
  year      = {2019}
}
```
```
@InProceedings{Mascharka_2018_CVPR,
author = {Mascharka, David and Tran, Philip and Soklaski, Ryan and Majumdar, Arjun},
title = {Transparency by Design: Closing the Gap Between Performance and Interpretability in Visual Reasoning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
} 
```
