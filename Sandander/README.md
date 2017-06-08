Santander Product Recommendation 
-----------------------

This competition is aiming to predict which products their existing customers will use in the next month based on their past behavior and that of similar customers. [here](https://www.kaggle.com/c/santander-product-recommendation).
The best single model we have obtained during the competition was an XGBoost model with multiclass booster of Public LB score 0.0301487 and Private LB score 0.0303696(Rank 124/1787 top7%).
----------------------

### Download the data

* Clone this repo to your computer.
* Get into the folder using `cd Santander`.
* Run `mkdir data`.
* Switch into the `data` directory using `cd data`.
* Download the data files from Kaggle into the `data` directory.  
    * You can find the data [here](https://www.kaggle.com/c/santander-product-recommendation/data).
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

* Switch to `feature` directory using `cd feature`
* Run `python creat_feature.py` to clean data set and generate new features.
    * This will replace the origanial train and test data in data folder.
* Switch to `model` dirctory using `cd` back to Santander and `cd model`
* Run `python xgb.py`.
    * This will dirctly train the xgboost model without CV, and predict the probability of each products
    . The final predict results are the 7 products with the highest probability.

