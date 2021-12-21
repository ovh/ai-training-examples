#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:10:47 2021

@author: epetton
"""

from flask import Flask

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return 'Web App with Python Flask using AI Training!'

if __name__ == '__main__':
    # starting app
    app.run(debug=True,host='0.0.0.0')