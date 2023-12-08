# ADL-CW

Data Preprocessing:
Before loading files in train.py, train_labels.pkl and val_labels.pkl need to be split into train_labels.pkl(0-9,a,b), validation.pkl(c) and val_labels.pkl(d,e,f).
I have a splitter.py which does this, however there are few things I did before running this script.

For samples
1. the previous file structure was as follows, train and val, 
2. In samples, I renamed val to valtest, so now valtest contains (d,e,f)
4. I then made a new folder called val.
3. Then from train, I moved the c directory into the empty val folder.
4. Now train contains (0-9,a,b), val contains (c) and valtest contains (d,e,f)


For annotations:
1. I created a new pkl file called validation.pkl
2. I then moved all files audio files that have part "a,b,c,d,e,f" in it to this new pkl

Once this is done there is a small problem where the file_path in val_labels.pkl starts with val/d, val/e, val/f
I ran a python script that went through each file and did a str.replace remove this so it just starts with d etc..

I also did the same for train_labels.pkl and validation.pkl all files that start with the part segement at the start of the files were removed using str.replace


Now the data is in the correct format, you will have to change the file paths to these datasets in my code. I have listed the steps below:

the dataset paths in the code need to be redirected to the path that is being used :

if the pickl files look something like this then:
MagnaTagATune/annotations/train_labels.pkl -train
MagnaTagATune/annotations/validation.pkl - val
MagnaTagATune/annotations/val_labels.pkl -test

and samples to look something like this 

/mnt/storage/scratch/gv20319/MagnaTagATune/samples/train --train
/mnt/storage/scratch/gv20319/MagnaTagATune/samples/val -- val
/mnt/storage/scratch/gv20319/MagnaTagATune/samples/valtest -- test



1. In lines 458 and 463 in the Trainer class 
2. change line 458 to add the MagnaTagATune/annotations/validation.pkl which is the validation pickle file containing parts c
3. change line 463 to add the MagnaTagATune/annotations/val_labels.pkll which is the test pickle file containing parts d,e,f


4. for lines 116, change the datasetpath to include the train_labels.pkl file, and change samplespath with /MagnaTagATune/samples/train
5. for lines 120, change the datasetpath to include the validation.pkl file, and change samplespath with /MagnaTagATune/samples/val
6.for lines 124, change the datasetpath to include the valtest.pkl file, and change samplespath with /MagnaTagATune/samples/train



if you want to test the extension part of the code add the --extension True 
for different lengths and strides use --length and --stride 

check the args parser for more parameters that can be added 
