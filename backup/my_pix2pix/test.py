import csv

arr = [["Data"]]

for i in range(5):
    arr.append(["{}".format(i)])

with open('losses.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(arr)