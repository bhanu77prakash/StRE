import sys
import numpy as np
from datetime import datetime, timedelta
from itertools import repeat
import xml.etree.ElementTree as ET
from collections import Counter
import re
import numpy as np
import difflib

if(len(sys.argv) != 3):
	print(len(sys.argv))
	sys.exit('Usage: python3 extractor.py <xml filename> <output filename>')
file_id = sys.argv[1]

# Start parsing the xml file.
tree = ET.parse(file_id)
root = tree.getroot()
NAMESPACE = "{http://www.mediawiki.org/xml/export-0.10/}"
# The edits is a list of tuples. First element in the tuple is the username and the second element is the dictionary that contains the user edit details i.e sha1, text, revision id, ...
edits = []
count_text = 0
for item in root.findall(".//"+NAMESPACE+"revision"):
	edit = {}
	for child in item:
		#	Under the contributor child there are three other children. So handled explicitly.
		if child.tag == NAMESPACE+'contributor':
			for grand in child:
				try:
					edit["<c>"+grand.tag] = grand.text.encode('utf-8')
					# print grand.tag
				except:
					edit["<c>"+grand.tag] = ''
		elif child.tag == NAMESPACE+'text':
			count_text+=1
			try:
				edit[child.tag] = child.text.encode('utf-8')
			except:
				edit[child.tag] = "".encode('utf-8')
		else:
			try:
				edit[child.tag] = child.text.encode('utf8')
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

file = open(sys.argv[2], "w", encoding="utf-8")
# for i in edits: # Uncomment this to get output for all edits in the file.

# i = edits[-1]
for i in range(len(edits)):
	# print(edits[i][1][NAMESPACE+'text'])
	file.write(edits[i][1][NAMESPACE+'text'].decode("utf-8")+"\n*$*$*$delimiter$*$*$*\n")
# file.write(edits[-1][1][NAMESPACE+'text'].decode("utf-8"))
