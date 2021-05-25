from sklearn.metrics import f1_score

op = open("ooooo.txt", "r")
dev = open("dev.txt", "r")

output = []
for i in op:
    single_line = i.rstrip('\n').split(" ")
    output.append(single_line)

devput = []
for i in dev:
    single_line = i.rstrip('\n').split(" ")
    devput.append(single_line)

for j in devput:
    print(j)
# f1 =  f1_score(output, devput, average = "macro")
# print(f1)