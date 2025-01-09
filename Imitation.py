import numpy as np
from numpy.random import choice, random , normal
import pygame
from colour import Color
"""
seed class repsent an entity that can draw with. it is have six featchers
(r,g,b,r,x,y) that used to draw a circle yes every 1d gene with six coloum
repsent a circle used to constract the target image
"""
class Seed:
    def __init__(self,genes):
        self.chrom = np.clip(genes,0,1)
        self.fitness = None
        self.pic_ref = None

    def __str__(self):
        return f'Fitness: {self.fitness} genes: {self.chrom.shape[0]}'

    def mutate(self,rate=0.01,add_chance=0.3,scale=0.3):

        chrom = np.copy(self.chrom)
        genes, feat = self.chrom.shape
        num_mutation = 1 + int(self.chrom.size * rate)

        if random() > add_chance:
            #we mutate a feachers in our 6 featchers
            #print(f'{choice(genes)} {choice(feat)}')
            #print(num_mutation)
            #print(choice([4,5]))
            scale = scale / num_mutation
            for i in range(num_mutation):
                if random() < 0.5:
                    chrom[choice(genes), 3] += normal() * scale
                else:
                    chrom[choice(genes), choice([4,5])] += normal() * scale
 
        # if we get random number less than add_chance that will
        # be rare and now we can add a new gene which is a new circle
        else:
            if random() > add_chance:
                chrom = np.append(chrom,random(size=(1,feat)), axis=0)
                #print(f'affter add {self.chrom}')
            else:
                chrom = np.delete(chrom , choice(genes), axis=0)

        return Seed(chrom)

    
    def crossover(self,other):
        
        copyme = np.copy(self.chrom)
        other_copy = np.copy(other.chrom)
        if copyme.shape == other_copy.shape:
            #we take the the color from copyme
            #and we take the postion x,y from other copy
            #print('in if')
            off_one = copyme[:,:4]
            off_tow = other_copy[:,-2:]
            res = np.append(off_one,off_tow, axis=1)
        else:
            # we add part from copyme and part from other_copy
            #print('in else')
            r1,c1 = copyme.shape
            r2,c2 = other_copy.shape

            dif = r1 - r2
            far = abs(dif)
            #'copyme' is greater
            if dif > 0:
                # 'other_copy' have genes less than 'far' and we cannot
                # slice it to index 'far' bc thes will skip from index 'far' to index 'dif'
                # in 'copyme'
                if far > r2:
                    p1 = other_copy[:,:]
                    p2 = copyme[r2:,:]
                else:
                    # other_copy have genes equals to 'far'
                    # and we can take the frist far form it
                    # and the other from 'copyme'
                    p1 = other_copy[:far,:]
                    p2 = copyme[far:,:]

            # other_copy is greater
            else:
                if far > r1:
                    p1 = copyme[:,:]
                    p2 = other_copy[r1:,:]
                else:
                    # i keep the slice even i skip genes in this case
                    p1 = copyme[:far,:]
                    p2 = other_copy[far:,:]

            res = np.append(p1,p2, axis=0)

        return Seed(res)
        

"""
poplutaion class repsent the world of 2d plane that seed object can live 
inside it we used pygame libary for this world
"""
class Population:

    def __init__(self,path):
        pygame.init()
        self.best_one = None
        self.pop = []
        #self.ref load target image into 3d array 
        self.ref = pygame.surfarray.pixels3d(pygame.image.load(path))
        w,h,d = self.ref.shape
        self.size = (w,h)
        self.window = pygame.display.set_mode(self.size)
        self.window.fill((255,255,255))

        #self.window.blit(self.image, (360,151))

    def __str__(self,pAll=False):
        re = ''
        if pAll:
            for o in self.pop:
                re += f'\n{o.__str__()}'
        return f'best Memeber {self.best_one} \n {re}'

    def draw(self,org):
        # clear the prev Seed object from the canvas
        self.window.fill((255,255,255))
        # take a new clean canvas
        screan = self.window.copy()
        for gene in org.chrom:
            pos = (gene[4]*self.size[0] ,gene[5]*self.size[1])
            #color = ((gene[0]*255), (gene[1]*255), (gene[2]*255))

            c = ((gene[0]), (gene[1]), (gene[2]))
            color = tuple(map(lambda x: int(255 * x),  Color(hsl=c).rgb))
            pygame.draw.circle(screan, color, pos, int((gene[3] * 0.3 + 0.01)* self.size[0]))
            #print(pos)
            #print(gene)
            #print(color)
        #print(org.chrom)
        return screan 
        
    def culc_fitness(self,seed):
        crr_snapshot_surfce = self.draw(seed)
        crr = pygame.surfarray.pixels3d(crr_snapshot_surfce)
        array_dif = crr - self.ref
        # this example elestrate the subtraction operation in 3darray
        # subtract the target image array from the crr orgnasim
        #print(f'{crr[55,55]} {self.ref[55,55]} {array_dif[55,55]}')
        #print(np.mean(np.abs(array_dif)))
        # high mean high difrence
        #sumGen = sum([i for i in range(seed.chrom.shape[0])])
        mean = np.mean(np.abs(array_dif))  + seed.chrom.shape[0]
        seed.fitness = (1 - (mean/(255)))*100
        seed.pic_ref = crr_snapshot_surfce 
        
    
    def mutate_and_pick(self, seed, rate, scale, add, tries=10):
        """mutate and make the seed more diverse"""
        for i in range(tries):
            o = seed.mutate(rate=rate, scale=scale, add_chance=add)
            self.culc_fitness(o)
            if o.fitness > seed.fitness:
                return o
        """ Mutate organism attempts times to try and get something better """
        return seed
    


    """
    create a population equals to 'pop_size' 
    etch seed object contains 'complexty' number of genes
    """
    def init(self,pop_size=50,complexty=10):
        for i in range(pop_size):
            crr = Seed(random((complexty,6)))
            self.culc_fitness(crr)
            self.pop.append(crr)
        #best fitness at index 0
        self.pop = sorted(self.pop, key=lambda x: -x.fitness)
        self.best_one = self.pop[0]



    def produse(self,mutation_rate=0.01):
        new_pop = []
        summ = sum([s.fitness for s in self.pop])
        #w = 1 - np.linspace(0,0.01,len(self.pop))
        length = len(self.pop)
        #summ = sum(range(length))
        #w = [i / summ for i in range(length)]
        w = [o.fitness / summ for o in self.pop]
        #print(w)
        #print(w/w.sum())
        for i in range(length):
            a,b = choice(self.pop,2 , replace=False, p=w)#p = w / w.sum())
            new_seed = a.crossover(b)
            #print(a,b)

            self.culc_fitness(new_seed)
            """
            if random() < mutation_rate:
                         new_seed = new_seed.mutate(rate=mutation_rate)
            """
            new_seed = self.mutate_and_pick(new_seed, mutation_rate ,0.1, 0.1,)

            #self.culc_fitness(new_seed)
            if new_seed.fitness > self.best_one.fitness:
                         self.best_one = new_seed
            #else:
                #cloning into new population
                #new_pop.append(self.best_one)

            new_pop.append(new_seed)

        self.pop = sorted(new_pop, key=lambda x: -x.fitness)
            
    
"""
the start of the program still in test vase
"""
def evolve(path,pop_size,Gen,rate=0.01):
    p = Population(path)
    p.init(pop_size)

    #print(p)
    for i in range(Gen):
        p.produse(mutation_rate = rate)
        p.window.blit(p.best_one.pic_ref ,(1,1))
        pygame.display.update()

        print(p.__str__(True))
        

# crossover Tests
"""
nrg = np.random.default_rng()
chrom = nrg.random((2,6))

p = Population('g.webp')
s = Seed(chrom)
o = Seed(nrg.random((6,6)))
re = s.crossover(o)

print(s.chrom)
print('_________________________________')
print(o.chrom)
print('_________________________________')
print(re.chrom)
"""
"""
# mutaion test
p = Population('g.webp')
nrg = np.random.default_rng()
chrom = nrg.random((2,6))
s = Seed(chrom)
p.best_one = s
while True:
    p.culc_fitness(s)
    if s.fitness > p.best_one.fitness:
        p.best_one = s
    print(p.best_one)
    
    print(s)
    p.window.blit(p.best_one.pic_ref ,(1,1))

    pygame.display.update()
    s = p.best_one.mutate(rate=0.06, add_chance = 0.3)

"""
evolve('./Mona_Lisa.webp', 10,50000,0.3)
