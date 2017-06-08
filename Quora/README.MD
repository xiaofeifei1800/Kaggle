 Quora Question Pairs
-----------------------

In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not.[here](https://www.kaggle.com/c/quora-question-pairs). 

The best single model we have obtained during the competition was an XGBoost model with binary classifier booster of Public LB score 0.14881 and Private LB score 0.15247(Rank 263/3396 top8%).

----------------------

### Download the data

* Clone this repo to your computer.
* Get into the folder using `cd Quora`.
* Run `mkdir data`.
* Switch into the `data` directory using `cd data`.
* Download the data files from Kaggle into the `data` directory.  
    * You can find the data [here](https://www.kaggle.com/c/quora-question-pairs/data).
    * You'll need to register with Kaggle and accept the agreement to download the data.
* Extract all of the `.zip` files you downloaded.
* Remove all the zip files by running `rm *.zip`.
* Switch back into the `Santander` directory using `cd ..`.

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt`.
    * Make sure you use Python 2.7.
    * You may want to use a virtual environment for this.

Usage
-----------------------

* Switch to `features` directory using `cd features`
* Run `python run_all.py` to generate new features.
    * This will run all the script in features directory by the order in `run_all.py`. The final features will be saved in 
    data folder.
* Switch to `model` dirctory using `cd` back to Quora and `cd models`
* Run `python xgb1.py`.
    * This will dirctly train the xgboost model and give the best single model results. The diiferent `xgb.py` and `xgb1.py` is using different way to do the updown sampling.
* Run `python temp.py`.
    * This will give you a stacking result from different models, I don't finish organize this script yet.
