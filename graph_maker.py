import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

counts = []
dates = []
nums = []
num = 1

with open('berkeley_graph_data.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        counts.append(row[0])
        if num % 200 == 0:
            dates.append(row[1])
        else:
            dates.append("")
        nums.append(num)
        num += 1

x = np.asarray(dates)
y = np.asarray(counts)

plt.figure(figsize=(15,5))
plt.xlabel("Times of Each Day")
plt.ylabel("Number of People")
plt.title("Number of People at Berkeley Gym at Different Times")
plt.xticks(nums,dates,rotation=90)
plt.plot(nums,counts,'-')
plt.show()
