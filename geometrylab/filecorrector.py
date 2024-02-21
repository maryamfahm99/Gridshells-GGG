# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:52:39 2019

@author: Davide
"""

import os

import fileinput

import copy

import re


# Does a list of files, and
# redirects STDOUT to the file in question


path = os.path.dirname(os.path.abspath(__file__))

file_name = path + '/geometry/meshpy_test.py'

def split(s, delim=' '):
    words = []
    word = []
    for c in s:
        if c not in delim:
            word.append(c)
        else:
            if word:
                words.append(''.join(word))
                word = []
    if word:
        words.append(''.join(word))
    return words

def fix_file(file_name):
    #file = open(file_name, encoding='utf-8')
    #for line in file:
    for line in fileinput.input(file_name, inplace = 1):
        words = split(line, ':,.()[]=+-* ')
        new_line = copy.copy(line)
        for word in words:
            n_word = copy.copy(word)
            c = 0
            for i in range(len(word)-1):
                first = word[0]
                prev = word[i]
                letter = word[i+1]
                #nex = word[i+2]
                if letter.isupper() and not first.isupper():
                    if prev.islower():
                        o_word = (n_word[:i+1+c] +
                                  '_' + letter.lower() +
                                  n_word[i+2+c:])
                        c += 1
                        new_line = new_line.replace(n_word, o_word, 1)
                        n_word = o_word
        print(new_line, end = '')


for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('filecorrector.py'):
            pass
        elif filename.endswith('.py'):
            file_name = os.sep.join([dirpath, filename])
            fix_file(file_name)


#fix_file(file_name)