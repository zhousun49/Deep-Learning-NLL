import ast 

f1 = open("dict1.txt", "r")
token1 = ast.literal_eval(f1.readline())
tag1 = ast.literal_eval(f1.readline())

f = open("dict.txt", "r")
token = ast.literal_eval(f.readline())
tag = ast.literal_eval(f.readline())

transversed1 = {}
for key, value in token1.items():
    transversed1[value] = key 

transversed = {}
for key, value in token.items():
    transversed[value] = key 

num_dif = 0
for i in range(1000):
    if transversed[i] != transversed1[i]:
        num_dif += 1
print("Number of different tokens: ", num_dif)

transversed1 = {}
for key, value in tag1.items():
    transversed1[value] = key 

transversed = {}
for key, value in tag.items():
    transversed[value] = key 

num_dif = 0
for i in range(56):
    if transversed[i] != transversed1[i]:
        num_dif += 1
print("Number of different tags: ", num_dif)