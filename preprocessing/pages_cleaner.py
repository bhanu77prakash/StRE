"""

   Code for cleaning the pages for which scores aren't generated. 
   The rest of the pages are used to train our models as well as the baseline models

"""


import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--f_scores', type=str, default='./',
                       help='Floder containing scores')
parser.add_argument('--f_pages', type=str, default='',
                       help='Folder containing pages')
args = parser.parse_args()

if (args.f_pages == ''):
	args.f_pages = args.f_scores

if(args.f_pages[-1] != '/'):
	args.f_pages+='/'

files = os.listdir(args.f_scores)
files = [x for x  in files if '.scores.txt' in x]
files = [x.split('.')[0] for x in files]

pages = os.listdir(args.f_pages)
pages = [x for x in pages if 'pages_' in x]
pages = [x for x in pages if '.scores.txt' not in x  and len(x.split('.')) == 1]  

remove = [x for x in pages if x not in files]
print("Removing " +str(len(remove))+" pages from "+args.f_pages+". Press CTRL-C to stop")
time.sleep(3)
for file in remove:
	print("Removing file " +args.f_pages+file)
	os.system('rm '+args.f_pages+file)
# print(len(pages))
print("Done")
