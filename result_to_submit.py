import numpy as np
import pandas as pd
import csv
from tqdm import tqdm


re_list = []
with open('./log/9_pre09.txt', 'r') as file:
    for line in file:
        # print(line)
        re_list.append(line.replace('\n', ''))

print(len(re_list))
print(re_list[0:1])


re_id = []
re_la = []
for line in re_list:
    line = line.strip().split(', ')
    id = line[0]
    re_id.append(id)

    num_0 = float(line[1])
    num_1 = float(line[2])
    if num_0 > num_1:
        label = 0
    else:
        label = 1
    re_la.append(int(label))
# print(re_id)
# print(re_la)

# df = pd.DataFrame({'id':re_id, 'label':re_la})
# print(df)
# df.to_csv('result.csv', index=False, sep=',')
# print('done!')

new_re = []
new_n = []
with open('./9_result09.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'label'])
    for i in tqdm(range(len(re_id))):
        new_re.append([re_id[i], re_la[i]])
        new_n.append([i, i+1])
        writer.writerow([re_id[i], re_la[i]])
    print('done!')


# with open('./result1.csv', 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['id', 'label'])
#     writer.writerows(new_re)
#     print('done!')

# with open('./result2.csv', 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['id', 'label'])
#     writer.writerows(new_n)
#     print('done!')

# tmp = np.array(new_re)
# np.savetxt('./result3.csv', tmp, delimiter=',')