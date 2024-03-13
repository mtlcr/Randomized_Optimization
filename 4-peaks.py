
import pandas as pd
import mlrose_hiive as mlrose
import time
import matplotlib.pyplot as plt

rdm_hill_climbing = []
genetic_algm = []
sim_annealing = []
MIMIC = []
bit_length_options = range(10,101,10)

for bit_length in bit_length_options:
    print(bit_length)
    problem = mlrose.DiscreteOpt(length=bit_length, fitness_fn=mlrose.FourPeaks(t_pct=.10), maximize=True, max_val=2)

    start_mark = time.time()
    RHC_curve = mlrose.random_hill_climb(problem, max_attempts=bit_length * 60, curve=True)
    end_mark = time.time()
    rdm_hill_climbing.append([bit_length, end_mark - start_mark, RHC_curve[2][-1, 0], RHC_curve[2][-1, 1], RHC_curve[2][-1, 1] / (end_mark - start_mark)])

    # good performing GA
    start_mark = time.time()
    GA_curve = mlrose.genetic_alg(problem,pop_size=10 * bit_length,mutation_prob=0.1, max_attempts= 10 * bit_length, curve= True)
    end_mark = time.time()
    genetic_algm.append([bit_length, end_mark - start_mark, GA_curve[2][-1, 0], GA_curve[2][-1, 1], GA_curve[2][-1, 1] / (end_mark - start_mark)])

    start_mark = time.time()
    SA_curve = mlrose.simulated_annealing(problem, schedule=mlrose.GeomDecay(decay=0.995, min_temp=0.5), max_attempts=10 * bit_length, curve=True)
    end_mark = time.time()
    sim_annealing.append([bit_length, end_mark - start_mark, SA_curve[2][-1, 0], SA_curve[2][-1, 1], SA_curve[2][-1, 1] / (end_mark - start_mark)])

    start_mark = time.time()
    MIMIC_curve = mlrose.mimic(problem, pop_size=500, max_attempts=5 * bit_length, curve=True)
    end_mark = time.time()
    MIMIC.append([bit_length, end_mark - start_mark, MIMIC_curve[2][-1, 0], MIMIC_curve[2][-1, 1],MIMIC_curve[2][-1, 1] / (end_mark - start_mark)])


#Bit string length vs. 3 KPIs
metrics = ["Bit String Length","Runtime","Fitness","Evaluation"]
plt.rcParams.update({'font.size': 16, 'font.family' : 'monospace'})

for i in range(1,4):
    df = pd.concat([pd.DataFrame(rdm_hill_climbing)[i],
                    pd.DataFrame(genetic_algm)[i],
                    pd.DataFrame(sim_annealing)[i],
                    pd.DataFrame(MIMIC)[i]], ignore_index=True, axis=1)
    df.index=bit_length_options
    df.columns = ["Random Hill Climbing","Genetic Algorithm","Simulated Annealing","MIMIC"]
    df.plot(marker='o',xlabel=metrics[0],ylabel=metrics[i], title=metrics[i] + " vs " + metrics[0],figsize=(10,7))
    plt.savefig(metrics[i] + " vs " + metrics[0],bbox_inches='tight')

df = pd.concat([pd.DataFrame(RHC_curve[2][:,0]),
            pd.DataFrame(GA_curve[2][:,0]),
            pd.DataFrame(SA_curve[2][:,0]),
            pd.DataFrame(MIMIC_curve[2][:,0])], ignore_index=True, axis=1)
df.columns = ["Random Hill Climbing","Genetic Algorithm","Simulated Annealing","MIMIC"]
df.plot(marker=".",xlabel="Iterations",ylabel="Fitness Value",
        title="Fitness Value vs Iterations",figsize=(10,7))
plt.savefig("Fitness Value vs Iterations", bbox_inches='tight')

plt.show()
