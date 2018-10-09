# MetagenomicVirusML

This repository contains the data and code to reproduce the results of "Machine Learning for detection of viral
sequences in human metagenomic datasets" (published in BMC Bioinformatics, https://link.springer.com/content/pdf/10.1186/s12859-018-2340-x.pdf).

All figures from the article are included as .png images, but can also be generated anew by running the ipython notebooks (They are made for Pyhton2.7, not Python3).

``random_forest_main.ipynb`` Main results, using leave-one-experiment-out cross validation

``random_forest_transfer-GenBank.ipynb`` Results when training a model on GenBank data and applying to data from metagenomic experiments. Notice the GenBank data is provided in zipped format to fit within GitHub's limit on file sizes, so you need to unpack it first.

## Artificial Neural Networks

Training neural networks with parameter values as in the article can be done by just running the LOO_2ORF.sh script. This script uses network_clean.py where the network is actually created, data read in, CV loop handled etc.

Unless you have a GPU, training the networks might take forever. So, we have also provided the output logs and predictions made by our networks in the "final_LOEO_cw" folder. Based on these prediction files, neural netowrks performance is summarized at the end ``random_forest_main.ipynb`` notebook.
