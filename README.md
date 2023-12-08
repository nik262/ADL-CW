# ADL-CW
Data Preprocessing:
Before loading files in train.py, train_labels.pkl and val_labels.pkl need to be split into new_train_labels.pkl(0-9,a,b), val_labels.pkl(c) and test_labels.pkl(d,e,f).
I have a splitter.py which does this, however there are few things I did before running this script.

For samples
1. the previous file structure was as follows, train and val, 
2. In samples, I renamed val to valtest, so now valtest contains (d,e,f)
4. I then made a new folder called val.
3. Then from train, I moved the c directory into the empty val folder.
4. Now train contains (0-9,a,b), val contains (c) and valtest contains (d,e,f)


For annotations:
1. I created a new pkl file called validation.pkl
2. I then moved all files audio files that have part in it to this new pkl

Once this is done there is a small problem where the file_path in test_labels.pkl starts with val/d, val/e, val/f
I ran a python script that went through each file and did a str.replace change this to valtest/d etc.. as it is in test.

Now the data is in the correct format, it can be passed in as an arguemnt in train.sh


IMP STEPS TO RUN THE ACTUAL TRAIN_AUDIO.py

the dataset paths in the code need to be redirected to the path that is being used :

if the pickl files look something like this then:
MagnaTagATune/annotations/train_labels.pkl
MagnaTagATune/annotations/validation.pkl
MagnaTagATune/annotations/val_labels.pkl

1. In lines 458 and 463 in the Trainer class 
2. change line 458 to add the MagnaTagATune/annotations/validation.pkl which is the validation pickle file containing parts c
3. change line 463 to add the MagnaTagATune/annotations/val_labels.pkll which is the test pickle file containing parts d,e,f


if you want to test the extension part of the code add the --extension True 
for different lengths and strides use --length and --stride 
