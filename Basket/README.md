nstacart Market Basket Analysis
-----------------------

This competition gives customer orders over time, and the aim of this competition is to predict if existing customers will order the same products which they order previously. [here](https://www.kaggle.com/c/instacart-market-basket-analysis).

The best model we have obtained during the competition was ensemble the top 4 best models we have with F1 score maximization of Public LB score 0.4056450 (Rank 110/2623 top4%) and Private LB score 0.4034573(Rank 189/2623 top8%).

----------------------

### Download the data

* Clone this repo to your computer.
* Get into the folder using `cd Basket`.
* Run `mkdir data`.
* Switch into the `data` directory using `cd data`.
* Download the data files from Kaggle into the `data` directory.  
    * You can find the data [here](https://www.kaggle.com/c/instacart-market-basket-analysis/data).
    * You'll need to register with Kaggle and accept the agreement to download the data.
* Extract all of the `.zip` files you downloaded.
* Remove all the zip files by running `rm *.zip`.
* Switch back into the `Basket` directory using `cd ..`.

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt`.
    * Make sure you use Python 3.0.
    * You may want to use a virtual environment for this.

Usage
-----------------------

* Run`R clean data.R` to proprocess the train and test data(I am sorry to mix with R and Python)
* Switch to `feature` directory using `cd feature`
* Run `python creat_feature.py` to generate new features.
    * This will replace the origanial train and test data in data folder.
* Switch to `model` dirctory using `cd` back to Santander and `cd model`
* Run `python xgb.py`.
    * This will dirctly train the xgboost model without CV, and predict the probability of each products
    . The final predict results are the 7 products with the highest probability.

