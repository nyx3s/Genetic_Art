import matplotlib.pyplot as plt
from Imitation import * 
import time
import os

def run_try(file):
    path = file
    gen = 50
    generation = 10000
    gen_iter = int(generation / gen)
    complexity = 15
    dir_sample = f'{path[0:-4]}_Evolution'
    all_fitness_list = []

    os.makedirs(f"./output/seeds/{dir_sample}", exist_ok = True)
    os.makedirs(f"./output/plots/{dir_sample}", exist_ok = True)

    p = Population(f'./Images/{path}')
    start = time.time()
    list_besties = ev_with_p(p,complexity,gen,0.01)
    pas = time.time() - start
    pygame.image.save(p.best_one.pic_ref, f'./output/seeds/{dir_sample}/Generation: {gen}_Time:{pas/60}_{p.best_one.fitness}.jpeg')
    for s in list_besties:
        all_fitness_list.append(s.fitness)

#print(len(all_fitness_list))
#input()
#list_besties = ev_with_p(p, complexity, gen, f'./output/seeds/{dir_sample}', start, 0.1)
#pygame.quit()
#fig, axis = plt.subplots()
#axis.plot(range(gen-1),[s.fitness for s in list_besties])
#plt.xlabel('Generation')
#plt.ylabel('Fitness Value')
#plt.savefig(f'./output/plots/{dir_sample}/{gen}_{path}.png',dpi=300,bbox_inches='tight')
#plt.show()
#print(pas/60,' menites')

    for i in range(2,gen_iter+1):
        seed = p.best_one
        p = Population(f'./Images/{path}')
        p.pop.append(seed)
        list_besties = ev_with_p(p,complexity,gen,0.01)
        pas = time.time() - start
        pygame.image.save(p.best_one.pic_ref, f'./output/seeds/{dir_sample}/Generation:{gen*i}_Time:{pas/60}_{seed.fitness}.jpeg')
        for s in list_besties:
            all_fitness_list.append(s.fitness)
#print(len(all_fitness_list))
    lp = sorted(all_fitness_list)
#print(len(lp))
#input()
    pygame.quit()
    fig, axis = plt.subplots()
    axis.plot(range(gen_iter*gen),lp)
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.savefig(f'./output/plots/{dir_sample}/{path}.png',dpi=300,bbox_inches='tight')

# old way
"""
    pygame.quit()
    fig, axis = plt.subplots()
    axis.plot(range(gen-1),[s.fitness for s in list_besties])
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.savefig(f'./output/plots/{dir_sample}/Gen: {i*gen}_{path}.png',dpi=300,bbox_inches='tight')
"""
# main 
file_names = ['Ban.jpeg','Mona_Lisa.jpg']
for f in file_names:
    run_try(f)
