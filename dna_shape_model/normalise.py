inps = [
    'data/sequences_8.fasta.EP',
    'data/sequences_8.fasta.HelT',
    'data/sequences_8.fasta.ProT',
    'data/sequences_8.fasta.MGW',
    'data/sequences_8.fasta.Roll',
    'data/sequences_rc_8.fasta.EP',
    'data/sequences_rc_8.fasta.HelT',
    'data/sequences_rc_8.fasta.ProT',
    'data/sequences_rc_8.fasta.MGW',
    'data/sequences_rc_8.fasta.Roll',
]

for inp in inps:
    out = inp + '.pre'
    count = 0
    with open(inp, "r") as f:
        with open(out, "w") as g:
            all_lists = []
            hi = -100000.0
            lo =  100000.0
            for line in f:
                if count % 3 == 0:
                    s = ''
                elif count % 3 == 1:
                    s = line.strip() + ','
                else:
                    s += line.strip()
                    l = [float(i) for i in s.split(',') if i != 'NA']
                    hi = max(max(l), hi)
                    lo = min(min(l), lo)
                    all_lists.append(l)
                count += 1
            print(hi, lo)
            for l in all_lists:
                ll = [2 * (x - lo)/ (hi - lo) - 1 for x in l]
                # print(l)
                # l = [str(x) for x in l]
                # print(l)
                print(','.join([str(x) for x in ll]), file=g)

            

