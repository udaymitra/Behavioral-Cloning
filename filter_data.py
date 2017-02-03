import sys
import csv
import numpy as np
import random

input_csv_file = sys.argv[1]
output_csv_file = sys.argv[2]

file = open(input_csv_file, 'r')
reader = csv.reader(file)
writer = csv.writer(open(output_csv_file, 'w'))

STEERING_BIN_SIZE = 0.05
MAX_COUNT_PER_BIN = 100

bin_examples_dict = {}
for row in reader:
    steer = float(row[4])
    normalized_steer = 1 + steer
    bin_num = int(normalized_steer / STEERING_BIN_SIZE)
    if bin_num in bin_examples_dict:
        bin_examples_dict[bin_num].append(row)
    else:
        bin_examples_dict[bin_num] = [row]

for bin, rows in bin_examples_dict.items():
    random.shuffle(rows)
    out_rows_in_bin = rows[:MAX_COUNT_PER_BIN]
    writer.writerows(out_rows_in_bin)