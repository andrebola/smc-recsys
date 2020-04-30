# MTLAB - Music Recommender Systems

This repository contains the code to generate recommendations using [LFM-360k dataset](https://www.upf.edu/web/mtg/lastfm360k)

# Install dependencies
To start create a virtualenv and install the dependencies:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

# Run Jupyter notebook

You can run the python notebook or you can run the scripts from the command line.
To run the notebook first install jupyter notebooks:

```
pip install jupyter
```

And then access in the browser to the url: `http://localhost:8888`. Note that you might need to enter the token that is in the console. After that open the notebook with name `Recsys_example.ipynb`.

# Compute recommendations only from command line

The following instructions are to get the results using the command line.

First run the script to make the split between train and test:
 - generate_mtrx.py

Then we have to run the training and evaluation with the script:
 - model_predict.py -d '{dimensions}' -f lastfm

We can see the similarity of the artists from the original data running the scirpt:
 - analyze_similars.py -f lastfm -a '{artist name}'
