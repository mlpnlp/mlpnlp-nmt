import sys
from collections import Counter

word_count = Counter()

threshold = 3
args = sys.argv
if len(args) >= 2:
    threshold = int(args[1])
sys.stderr.write('threshold = {}\n'.format(threshold))

for line in sys.stdin:
    line = line.strip("\n")
    words = line.split()
    for word in words:
        word_count[word] += 1

for word, num in word_count.most_common():
    if num < threshold:
        break
    print(word + "\t" + str(num))
