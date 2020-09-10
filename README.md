# clevr-refplus

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
