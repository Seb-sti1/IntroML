# 02450 - Introduction to Machine Learning and Data Mining


## Install & Start

In the root folder of the repository,

* To install, run the following

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

* To start, run the following

```sh
source .venv/bin/activate
PYTHONPATH="$PYTHONPATH:${pwd}" python intro_ml/main.py
```

*Note: if you're using PyCharm, it updates the `PYTHONPATH` automatically, so you
can just create a python task for `main.py` and run it.*

## Dataset

The dataset used here is available on [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/109/wine)