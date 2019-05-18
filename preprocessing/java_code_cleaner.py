import re,os
import sys
from subprocess import check_output
import multiprocessing

if(len(sys.argv)!= 3):
    print("Usage: python3 java_code_cleaner.py text_extracted_file output_file>")
    exit()


file = open(sys.argv[1], "r")
file2 = open(sys.argv[2], "w")

line = file.read()
lines = line.split("\n*$*$*$delimiter$*$*$*\n")

processes = 48



# line = file.readlines()

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
    # Section wise remover.
    # final = re.sub('==References==.*?==', '==', final)
    # final = re.sub('==Notes==.*?==', '==', final)
    # final = re.sub('==External==.*?==', '==', final)

    final = re.sub('====.*?====', ' ', final)
    final = re.sub('===.*?===', ' ', final)
    final = re.sub('==.*?==', ' ', final)
    return final
    # file2.write(re.sub(r'TEMPLATE\[.*?\]', '', i)))



# for i in lines:
#     if(i.strip() !=''):
#         temp_file = open(sys.argv[2]+".temp_file", "w")
#         temp_file.write(i)
#         temp_file.close()
#         out = check_output(["java", "-cp", "de.tudarmstadt.ukp.wikipedia.jar", "de.tudarmstadt.ukp.wikipedia.tutorial.parser.T1_SimpleParserDemo", sys.argv[2]+".temp_file"])
#         # print(out.decode('utf-8'))
#         final = cleaner(out.decode('utf-8'))
#         file2.write(final+"\n*$*$*$delimiter$*$*$*\n")
#     else:
#         file2.write(" \n*$*$*$delimiter$*$*$*\n")


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

# print(return_dict[0])
for i in range(len(lines)):
    file2.write(return_dict[i])