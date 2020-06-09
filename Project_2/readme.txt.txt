#Rhea Sharma
#1001679252

SYSTEM REQUIREMENTS:

- Windows OS
- Python 3.x installed
- Python libraries required: os, math, random, time

To test the preprocessing.py file:

1) Delete the existing directories: './20_newsgroups' and './train_data'.
2) Download and unzip the data in the directory.(http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes/20_newsgroups.tar.gz)
3) Run 'python3 preprocessing.py'

This will create a directory named 'train_data', which will have training data.

Do not execute the preprocessing.py file without deleting  './20_newsgroups' and './train_data' as it will throw an error because the folders already exist.

After the data has been partitioned, execute 'python3 main.py'