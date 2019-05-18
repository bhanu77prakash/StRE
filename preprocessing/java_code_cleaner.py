# ===========================================================================================
# =           Code to convert xml format of the text tag to the plain text format           =
# ===========================================================================================



import re,os
import sys
from subprocess import check_output
import multiprocessing

if(len(sys.argv)!= 3):
    print("Usage: python3 java_code_cleaner.py <text_extracted_file> <output_file>")
    exit()


file = open(sys.argv[1], "r")
file2 = open(sys.argv[2], "w")

# ==================================================================
# =           Read the data from the extracted text data           =
# ==================================================================

line = file.read()
lines = line.split("\n*$*$*$delimiter$*$*$*\n")

processes = 48


# ======  End of Read the data from the extracted text data  =======


# ================================================================================
# =           Removes the links, additional markup formats in the data           =
# ================================================================================

def cleaner(i):
# for i in line:
    i = i.strip()
    l = len(i)
    final = ""
    j = 0
    while(j<l):
        if(i[j:min(j+9, l)] == 'TEMPLATE['):
            count = 1
            j = j+9
            while(count>0 and j<l):
                if(i[j] == '['):
                    count+=1
                elif(i[j] == ']'):
                    count-=1
                j+=1
            final+=' '
        elif(i[j:min(j+2, l)] == '[['):
            count = 1
            j = j+2
            while(count>0 and j<l):
                if(i[j:min(j+2,l)] == '[['):
                    count+=1
                elif(i[j:min(j+2,l)] == ']]'):
                    j+=1
                    count-=1
                j+=1
            final+=' '
        elif(i[j:min(j+2, l)] == '{|'):
            count = 1
            j = j+2
        # print(str(j)+ " "+str(l)+"\n")
            while(count>0 and j<l):
                if(i[j:min(j+2,l)] == '{|'):
                    count+=1
                elif(i[j:min(j+2,l)] == '|}'):
                    j+=1
                    count-=1
                j+=1
            final+=' '
        else:
            final+=i[j]
            j+=1
    # print("Finished File\n")
    final = re.sub('==See also==.*', '==', final)
    final = re.sub('Links:.*', ' ', final)

    final = re.sub('====.*?====', ' ', final)
    final = re.sub('===.*?===', ' ', final)
    final = re.sub('==.*?==', ' ', final)
    return final

# ======  End of Removes the links, additional markup formats in the data  =======


# =========================================================================
# =           Multi processing code to run the file in parallel           =
# =========================================================================
# The java file needs to be called on single edit and hence each edit is passed individually by storing in a plain text file. 
# The .temp file stores the text of a single edit. This is passed as input to the jar file.

def processor(string, index, return_dict):
    if(string.strip() !=''):
        temp_file = open(sys.argv[2]+"."+str(index)+"."+".temp_file", "w")
        temp_file.write(string)
        temp_file.close()
        out = check_output(["java", "-cp", "de.tudarmstadt.ukp.wikipedia.jar", "de.tudarmstadt.ukp.wikipedia.tutorial.parser.T1_SimpleParserDemo", sys.argv[2]+"."+str(index)+"."+".temp_file"])
        # print(out.decode('utf-8'))
        final = cleaner(out.decode('utf-8'))
        return_dict[index] = final+"\n*$*$*$delimiter$*$*$*\n"
    else:
        return_dict[index] = " \n*$*$*$delimiter$*$*$*\n"

    os.system("rm "+sys.argv[2]+"."+str(index)+"."+".temp_file")

# ======  End of Multi processing code to run the file in parallel  =======


# ===================================================================================================================
# =           Multi Processing code to process the edits in parallel. Each edit is processed individually           =
# ===================================================================================================================

manager = multiprocessing.Manager()
return_dict = manager.dict()

for i in range(0, len(lines), processes):
    jobs=[]
    for j in range(min(processes, len(lines)-i)):
        p = multiprocessing.Process(target=processor, args=(lines[i+j], i+j, return_dict, ))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

# Stores the processed text in the given file directory.
for i in range(len(lines)):
    file2.write(return_dict[i])

# ======  End of Multi Processing code to process the edits in parallel. Each edit is processed individually  =======
