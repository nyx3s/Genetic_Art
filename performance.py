import matplotlib.pyplot as plt
from Imitation import * 
import time

path = 'haf.jpg'
gen = 50
complexity = 15

p = Population(f'./Images/{path}')
start = time.time()
list_besties = ev_with_p(p,complexity,gen,0.01)
pas = time.time() - start
pygame.image.save(p.best_one.pic_ref, f"./output/seeds/{path}_{gen}_{pas}.jpeg")
pygame.quit()
fig, axis = plt.subplots()
axis.plot(range(gen-1),[s.fitness for s in list_besties])
plt.savefig(f'./output/{path}.png',dpi=300,bbox_inches='tight')
#plt.show()
print(pas/60,' menites')
while True:
    seed = p.best_one
    p = Population(f'./Images/{path}')
    p.pop.append(seed)
    list_besties = ev_with_p(p,complexity,gen,0.01)

