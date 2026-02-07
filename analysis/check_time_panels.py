import json,statistics
from math import sqrt
pfile='results/experiment_results_1767414768.json'
try:
    with open(pfile) as f:
        data=json.load(f)
except Exception as e:
    print('ERROR loading JSON:', e)
    raise
by_p={}
for r in data['results']:
    p=r['p']
    by_p.setdefault(p,[]).append(r['history'])

print('Loaded p levels:', sorted(by_p.keys()))
for p in sorted(by_p.keys()):
    try:
        reps=by_p[p]
        n=len(reps)
        t=len(reps[0]['hamming_distance'])
        final_vals=[rep['hamming_distance'][-1] for rep in reps]
        mean_final=sum(final_vals)/n
        # compute mean trajectory and record first/last
        means=[sum(rep['hamming_distance'][i] for rep in reps)/n for i in range(t)]
        sem_final=(statistics.pstdev(final_vals)/sqrt(n))
        print(f"p={p:.2f}: n={n}, final_mean={mean_final:.3f}, final_sem={sem_final:.3f}, first_mean={means[0]:.3f}, last_mean={means[-1]:.3f}")
    except Exception as e:
        print(f'ERROR processing p={p}:', e)

# Print detailed sample for p=0.75 and p=0.90
for q in (0.75,0.9):
    if q in by_p:
        reps=by_p[q]
        n=len(reps)
        t=len(reps[0]['hamming_distance'])
        means=[sum(rep['hamming_distance'][i] for rep in reps)/n for i in range(t)]
        print('\nSample means for p=',q)
        for idx in range(0,t,10):
            gen=idx*10
            print(f'gen {gen:>4}: mean {means[idx]:.3f}')
