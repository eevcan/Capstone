# Data Science Nanodegree Program by Udacity. 
Capstone Project

## Project Overview

This Project started by a personal interest, the TCG (Trading Card Game) is getting immensly big,
a lot of Youtubers and more propably played a huge part in this.
my foxus went onto magic the Gathering, while it maybe doesnt have the biggest data sets, it stil contains some pretty hefty prices
and i looked at those.


## Quick Start
    bash:
    0. extract datas.zip into "cardcosts\data" here will be a extract_here.txt
    1. cd (to the folder that was just downloaded)
    2. the data is already cleaned via "python data_analyse_modeling.py"          
    
    ->
    2. python generate_questions.py (i already generated about 1000 questions, the more questions the m,ore acurate but due to time issues i only did 1000 to prove)
    3. python train_model.py (i did not include a file since its very big and cant be uploaded)
    3. python app.py
    4. go to http://127.0.0.1:5000/

    Now i have cleaned data a beginner Machine Learning tool that analyses user input data on a website and gives u proper responses
    this is just to prove it works, due to time isses (this project has to be approved by trhe end of march)

## what changed in the ML code (improvements)

    Model 1 was unacurate and didnt recognize upper and lower cases
    Model 2 was slow and didnt undersatand it when i rephrased sentences like : whats the weather today -> today wheater whats it like
    Model 3 -> didnt improve as i wanted
    Model 4 (active) This one was when i found out About sentence transformer, this made the plk learning process faster and immensly accurate



## Example Questions to ask:
        whats the manacost of angel of mercy
        whats the number of angel of mercy
        rarity of angel of mercy
        angel of mercy Originaltext
        manaCost from angel of mercy
        What is the type of Ancestor's Chosen?

# Blog Post to share my new insights
    
  ## https://medium.com/@eevcoh/the-rising-price-of-trading-cards-bfed6966aab7


# Information about my Research

    I’ve started by using a free dataset from a TCG website, focusing on MTG because it’s one of the longest existing. Also, to include every TCG, I would need hundreds of GB of data.
    I’ve started by cleaning the data, but since it’s professional data, it’s already cleaned. However, I still left the cleaning in the code for future possibilities.
    
    Then, I began merging datasets to better understand the data. I could check for factors that explain why cards A, B, and C are so expensive.
    I created some graphs for further understanding and then wrote my blog post.
    
    In the future, I want to implement automatic data refreshes, but this wouldn’t be possible at the moment since my knowledge has reached a ceiling for now. I would need to get in contact with official websites.

    I hope this could awaken some interest
