import numpy as np
import matplotlib.pyplot as plt

reverts = np.load("reverted_edits.npy")
file = open("output.txt", "r")
lines = file.readlines()
lines = lines[1:]
lines = [x.strip().split("#") for x in lines]

normal = []
revert = []
print(len(reverts))
# print(type(lines[0][0]))
for i in range(len(lines)):
	if(int(lines[i][0]) in reverts):
		revert.append(float(lines[i][4]))
	else:
		normal.append(float(lines[i][4]))

print("scored >= 0 in normal "+str(len([x for x in normal if x >= 0]))+" / " +str(len(normal)))
print("scored >= 0 in reverted "+str(len([x for x in revert if x >= 0]))+" / " +str(len(revert)))
print("scored < 0 in normal "+str(len([x for x in normal if x < 0]))+" / " +str(len(normal)))
print("scored < 0 in reverted "+str(len([x for x in revert if x < 0]))+" / " +str(len(revert)))


# print(len([x for x in revert if x >= 0]))
# print(len([x for x in normal if x < 0]))
# print(len([x for x in revert if x < 0]))
# print(min(normal), max(normal))
# print(min(revert), max(revert))
# print(normal)
# num_bins = 20
plt.hist(normal, bins = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.show()