# Import packages
import streamlit as st
import numpy as np
import re
import nltk #natural language toolkit
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

dataset = "/Users/gloriabotchway/PycharmProjects/PythonProject/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
with open(dataset, "r", encoding="UTF-8") as file:
    lines = file.readlines()
