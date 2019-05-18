# ==================================================================================
# =           Code to convert the raw data form to the required format.            =
# ==================================================================================


# Parses two files in parallel the extracted text file and the original text file

import sys
import numpy as np
from datetime import datetime, timedelta
from itertools import repeat
import xml.etree.ElementTree as ET
from collections import Counter
import re
import numpy as np
import difflib
from nltk import tokenize
import multiprocessing,os
from tqdm import tqdm
import atexit
import sys
import time
import fuzzyset
from Levenshtein import distance
from fuzzywuzzy import fuzz

processes = 64
minprocesses = 8


if(len(sys.argv) != 6):
    print(len(sys.argv))
    sys.exit('Usage: python parser.py <xml filename> <temporary file to store some data(can be removed)> <scores file> <cleaned verion of page(output of java_cleaner.py)> <final output file>')
file_id = sys.argv[1]

file = open(sys.argv[3], "r")
lines = file.readlines()
file.close()
lines = lines[1:]
lines = [x.strip().split("#") for x in lines]
lines = np.array(lines)
file = open(sys.argv[2], 'w')

if(len(lines)<5):
    exit()

# ===========================================================================================
# =           Removes the parenthesis part and keeps plain text in markup format           =
# ===========================================================================================

def parenthesis(text):
    flag = 0
    flag1 = 0
    final_text = ""
    count = 0
    for i in text:
        if(i == '{' and flag == 0):
            final_text+=i
            flag = 1
            pass
        elif(i == '{' and flag == 1):
            final_text = final_text[:-1]
            count+=1
            flag = 0
        elif(i == '}' and flag1 == 0):
            final_text+=i
            flag1 = 1
            pass
        elif(i == '}' and flag1 == 1):
            final_text = final_text[:-1]
            count-=1
            flag1=0
        elif(count == 0):
            final_text+=i
    return final_text

# ======  End of Removes the parenthesis part and keeps plain text in markup format  =======

# ===============================================================
# =           Removes the ending sections in the page           =
# ===============================================================

def parse_wiki(text):
    text = parenthesis(text)
    text = re.sub("==References==(.|\n)*", "", text)
    text = re.sub("==External links==(.|\n)*", "", text)
    text = re.sub("==.*==|\[\[|\]\]|<.*?>([^<].|\n)*<.*?>|'''|\*|<.*?>", "", text)
    text = re.sub("\[http://.*?\]", "", text)
    return text.strip()

# ======  End of Removes the ending sections in the page  =======


# This part of the code takes input file name

# Start parsing the original xml file.
tree = ET.parse(file_id)
root = tree.getroot()
NAMESPACE = "{http://www.mediawiki.org/xml/export-0.10/}"
# The edits is a list of tuples. First element in the tuple is the username and the second element is the dictionary that contains the user edit details i.e sha1, text, revision id, ...
edits = []
count_text = 0
for item in root.findall(".//"+NAMESPACE+"revision"):
    edit = {}
    for child in item:
        #    Under the contributor child there are three other children. So handled explicitly.
        if child.tag == NAMESPACE+'contributor':
            for grand in child:
                try:
                    edit["<c>"+grand.tag] = grand.text
                    # print grand.tag
                except:
                    edit["<c>"+grand.tag] = ''
        elif child.tag == NAMESPACE+'text':
            count_text+=1
            try:
                edit[child.tag] = parse_wiki( child.text)
            except:
                edit[child.tag] = ""
        else:
            try:
                edit[child.tag] = child.text
            except:
                edit[child.tag] = ''
    # inserting username into first element of the tuple.
    try:
        edits.append((edit['<c>'+NAMESPACE+'username'], edit))
    except:
        try:
            edits.append((edit['<c>'+NAMESPACE+'ip'], edit))
        except:
            edits.append(("RANDOM_USER_TAG", edit))


count = 0
final_edits = []

# =============================================================
# =           Handling the cleaned text in parallel           =
# =============================================================

redirect_file = open(sys.argv[4], "r")
line = redirect_file.read()
line = line.strip().split("*$*$*$delimiter$*$*$*")
line = line[:-2]
for i in range(len(edits)):
    edits[i][1][NAMESPACE+'text'] = line[i]

# Removing continuous edits by same user as explained in the paper.
for i in range(0,len(edits)):
    try: 
        if(edits[i][0] == edits[i+1][0]):
            count+=1
        else:
            final_edits.append(edits[i])
            count = 0
    except:
        final_edits.append(edits[i])
        count = 0    

# Storing the content in a temporary file where edits are stored along with the revision id's
count = 0
for i in final_edits:
    if(i[1][NAMESPACE+'id'] in lines[:, 0]):
        count+=1
        file.write(i[1][NAMESPACE+'text']+"\n?==?*delimiter*?==?"+lines[list(lines[:, 0]).index(i[1][NAMESPACE+'id'])][4]+"?==?\n")

file.close()

temp = []
for i in final_edits:
    # If we have a valid score for this revision ID then process the edit (edits are stored in form of sentence level tokens)
    # If needed other tags, please append to this tuple at the end and accordingly update the sentence format to be stored in the final file
    if(i[1][NAMESPACE+'id'] in lines[:, 0]):
        temp.append((tokenize.sent_tokenize(i[1][NAMESPACE+'text']), lines[list(lines[:, 0]).index(i[1][NAMESPACE+'id'])][4]))

# Removing the extra spaces and new line characters from the sentences
for i in range(len(temp)):
    for j in range(len(temp[i][0])):
        temp[i][0][j] = temp[i][0][j].strip()

# Removing the null tokens from the list of tokens in the edit
lines = []
for i in temp:
    line = []
    for j in i[0]:
        if j != '':
            line.append(j)
    if(len(line)!=0):
        lines.append((line, i[1]))

# Identifying which sentences are the changes in the edit and the previous version of it.
indices = []
for i in range(1, len(lines)):
    indices.append([x for x in range(len(lines[i][0])) if lines[i][0][x] not in lines[i-1][0]])
    indices[-1] += [x for x in range(len(lines[i-1][0])) if (lines[i-1][0][x] not in lines[i][0])]


changes = []

# for i in range(len(indices)):
#     for j in range(len(indices[i])):
#         if(indices[i][j] in range(len(lines[i+1][0]))):
            
#             # print len(difflib.get_close_matches(lines[i+1][0][indices[i][j]], [""]+lines[i][0])), len(lines[i][0]), len(lines[i+1][0])
#             print("entered")
#             a = fuzzyset.FuzzySet()
#             for stri in lines[i][0]:
#                 a.add(stri)
#             prev_version = a.get(lines[i+1][0][indices[i][j]])[0][1]
#             print("exited")
#             # prev_version = difflib.get_close_matches(lines[i+1][0][indices[i][j]], lines[i][0], cutoff = 0.0)[0]
#             if(distance(prev_version,lines[i+1][0][indices[i][j]]) > distance('',lines[i+1][0][indices[i][j]])):
#             # if(len([li for li in list(difflib.ndiff(prev_version,lines[i+1][0][indices[i][j]])) if li[0] != ' ']) > len([li for li in list(difflib.ndiff('',lines[i+1][0][indices[i][j]])) if li[0] != ' '])):
#                 prev_version = ''
#             changes.append((lines[i+1][0][indices[i][j]]+" #$#$# "+prev_version, lines[i+1][1]))
#         else:
#             changes.append((''+" #$#$# "+lines[i][0][indices[i][j]], lines[i+1][1]))
#         print("Finised Iter")


# =========================================================================================================
# =           function that matches the sentence that is previous vesion of a modified sentence           =
# =========================================================================================================
# This function uses fuzzyset function to identify the nearest sentence that this sentence is modified from

def processor(index, return_dict):
    temp_list = []
    for ja in range(len(indices[index])):
        if(indices[index][ja] in range(len(lines[index+1][0]))):
            a = fuzzyset.FuzzySet()
            for stri in lines[index][0]:
                a.add(stri)
            if(a.get(lines[index+1][0][indices[index][ja]]) != None):
                prev_version = a.get(lines[index+1][0][indices[index][ja]])[0][1]
            else:
                prev_version = ''
            # prev_version = difflib.get_close_matches(lines[index+1][0][indices[index][ja]], lines[index][0], cutoff = 0.0)[0]
            if(distance(lines[index+1][0][indices[index][ja]], prev_version) >= distance(lines[index+1][0][indices[index][ja]], '')):
            # if(len([li for li in list(difflib.ndiff(prev_version,lines[index+1][0][indices[index][ja]])) if li[0] != ' ']) > len([li for li in list(difflib.ndiff('',lines[index+1][0][indices[index][ja]])) if li[0] != ' '])):
                prev_version = ''
            if(fuzz.ratio(prev_version, lines[index+1][0][indices[index][ja]])<75):
                prev_version = ''
            temp_list.append(((lines[index+1][0][indices[index][ja]]+" #$#$# "+prev_version, lines[index+1][1])))
        else:
            temp_list.append(((''+" #$#$# "+lines[index][0][indices[index][ja]], lines[index+1][1])))

    return_dict[index] = temp_list

# ======  End of function that matches the sentence that is previous vesion of a modified sentence  =======

    

# def processor1(index, ja, return_dict1):
#     if(indices[index][ja] in range(len(lines[index+1][0]))):
#         # print("someting")
#         a = fuzzyset.FuzzySet()
#         for stri in lines[index][0]:
#             a.add(stri)
#         if(a.get(lines[index+1][0][indices[index][ja]]) != None):
#             prev_version = a.get(lines[index+1][0][indices[index][ja]])[0][1]
#         else:
#             prev_version = ''
#         # prev_version = difflib.get_close_matches(lines[index+1][0][indices[index][ja]], lines[index][0], cutoff = 0.0)[0]
#         if(distance(lines[index+1][0][indices[index][ja]], prev_version) >= distance(lines[index+1][0][indices[index][ja]], '')):
#         # if(len([li for li in list(difflib.ndiff(prev_version,lines[index+1][0][indices[index][ja]])) if li[0] != ' ']) > len([li for li in list(difflib.ndiff('',lines[index+1][0][indices[index][ja]])) if li[0] != ' '])):
#             prev_version = ''
#         if(fuzz.ratio(prev_version, lines[index+1][0][indices[index][ja]])<75):
#             prev_version = ''
#         return_dict1[ja] = ((lines[index+1][0][indices[index][ja]]+" #$#$# "+prev_version, lines[index+1][1]))
#     else:
#         return_dict1[ja] = ((''+" #$#$# "+lines[index][0][indices[index][ja]], lines[index+1][1]))

# def processor(index, return_dict):
#     # print(index)
#     temp_list = []
#     # os.system("touch "+str(index)+".txt")
#     manager1 = multiprocessing.Manager()
#     return_dict1 = manager1.dict()
#     jobs1 = []
#     for ja in range(0, len(indices[index]), minprocesses):
#         for j in range(min(minprocesses, len(indices[index])-ja)):
#             count = 0
#             for proc in jobs1:
#                 if(proc.is_alive() == True):
#                     count+=1
#                 else:
#                     proc.join()
#             while (count == minprocesses):
#                 count = 0
#                 for proc in jobs1:
#                     if(proc.is_alive() == True):
#                         count+=1
#                     else:
#                         proc.join()
#             p = multiprocessing.Process(target=processor1, args=(index, ja+j, return_dict1))
#             jobs1.append(p)
#             p.start()
#     # print("Enter for " + str(index))
#     for proc in jobs1:
#         proc.join()
#     # print("Exit for "+str(index))
#     for i in range(len(indices[index])):
#         temp_list.append(return_dict1[i])

#     return_dict[index] = temp_list
#     # os.system("rm "+str(index)+".txt")

    

# ========================================================================
# =           Multi Process handling of the edits in parallel.           =
# ========================================================================

manager = multiprocessing.Manager()
return_dict = manager.dict()

for i in range(len(indices)):
    return_dict[i] = []


for i in tqdm(range(0, len(indices), processes)):
    jobs=[]
    for j in range(min(processes, len(indices)-i)):
        
        p = multiprocessing.Process(target=processor, args=(i+j, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

# ======  End of Multi Process handling of the edits in parallel.  =======


count = 0
for i in range(len(indices)):
    for j in range(len(return_dict[i])):
        changes.append(return_dict[i][j])

# Store the sentences in the following format in the file
file2 = open(sys.argv[5] ,'w')
for i in changes:
    file2.write(i[0] + "\n####SCORE###\n"+str(i[1])+"\n####SCORE###\n")
