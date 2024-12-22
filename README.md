# cs383
Final project repository for CS383 Machine learning class

To replicate results from the paper:
Figure 1. TF-IDF vs Bag of words
```python main.py -d combined -f tdif -c no```
```python main.py -d combined -f bow -c no```

Figure 2. Classification reports for all cleanup types
For no cleanup data. python main.py -d combined -f tdif -c no
For simple cleanup data. python main.py -d combined -f tdif -c simple
For spelling cleanup data. NOTE: it takes approximately 10-30 minutes for the spelling fixes to complete.  python main.py -d dev_only -f tdif -c spell
For manual cleanup. python main.py -d dev_cleaned -f tdif -c manual

Figure 3 and 4
python main.py -d combined -f tdif -c no
When the first matrix for Logistic regression is displayed, close the plot window and the second matrix for SVM will show up
