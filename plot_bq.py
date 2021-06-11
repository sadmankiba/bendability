from bq import find_bq 
from read_data import get_processed_data

import matplotlib.pyplot as plt
import numpy as np

UNIT_NUCLEOTIDE_SIZE = 5

(cnl_df, rl_df, tl_df, chrvl_df, libl_df) = get_processed_data()

seq_bq_map = find_bq(cnl_df, unit_size=UNIT_NUCLEOTIDE_SIZE)

sorted_unit_seq_bq_pair = sorted(seq_bq_map.items(), key=lambda x: x[1])
print('sorted_unit_seq_bq_pair\n', sorted_unit_seq_bq_pair)

# Plot
if len(sorted_unit_seq_bq_pair) > 20:
    pairs_to_show = [ pair for (i, pair) in enumerate(sorted_unit_seq_bq_pair) if i < 10 or i >= len(sorted_unit_seq_bq_pair) - 10 ]
else:
    pairs_to_show = sorted_unit_seq_bq_pair

print('pairs to show\n', pairs_to_show)

x = [ pair[0] for pair in pairs_to_show]
y = [ pair[1] for pair in pairs_to_show]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Move bottim x-axis to centre
ax.spines['bottom'].set_position(('data', 0))

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.bar(x, y)
plt.ylabel('Bendability quotient')
plt.setp(ax.get_xticklabels(), rotation=90, va='top')
plt.savefig('figures/cnl_bq_5.png')

plt.show()