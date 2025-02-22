import numpy as np
from numpy.random import choice, random , normal
import pygame
from colour import Color
#import colour as cl
#from math import log10, sqrt
from PIL import Image
"""
seed class repsent an entity that can draw with. it is have six featchers
(h,s,l,a,r,x,y) that used to draw a circle yes every 1d gene with six coloum
repsent a circle used to constract the target image
"""
class Seed:
    def __init__(self,genes):
        self.chrom = np.clip(genes,0,1)
        self.fitness = None
        self.pic_ref = None

    def __str__(self):
        return f'Fitness: {self.fitness} genes: {self.chrom.shape[0]}'
    """
    @pram 1:mutation rate , 2:probabilty of adding new 1d genes ,3:scale factoer
    @return a new mutation version of seed object
    """
    def mutate(self,rate=0.01,add_chance=0.1,scale=0.1):

        chrom = np.copy(self.chrom)
        genes, feat = self.chrom.shape
        num_mutation = 1 + int(self.chrom.size * rate)

        if random() > add_chance:
            #we mutate a feachers in our 6 featchers
            #print(f'{choice(genes)} {choice(feat)}')
            #print(num_mutation)
            #print(choice([4,5]))
            #print(choice(4))
            scale = scale / num_mutation
            for i in range(num_mutation):
                if random() < 0.7:
                    chrom[choice(genes), choice([0,3])] += normal() * scale 
                else:
                    chrom[choice(genes), choice([1,2,4,5,6])] += normal() * scale
 
        # if we get random number less than add_chance that will
        # be rare and now we can add a new gene which is a new circle
        else:
            if random() > add_chance:
                chrom = np.append(chrom,random(size=(1,feat)), axis=0)
                #print(f'affter add {self.chrom}')
            else:
                chrom = np.delete(chrom , choice(genes), axis=0)

        return Seed(chrom)

    """
    @pram other: seed object to mate with
    @return new offspring of seed object that share the both genes in tow way
    @way one: if both have equal genes aka 'rows' both seeds will share there 6ix featchers
    @way tow: if not: we share only genes aka 'rows' of and return a new seed object length
    equal to the seed with longest row
    """ 
    def crossover(self,other):
        
        copyme = np.copy(self.chrom)
        other_copy = np.copy(other.chrom)
        res = []
        if copyme.shape == other_copy.shape:
            #we take the the color from copyme
            #and we take the postion x,y from other copy
            #print('in if')
            for i in range(copyme.shape[1]):
                if random() > 0.6:
                    s = copyme[:,i]
                else:
                    s = other_copy[:,i]
                res.append(s)
                #print(res)
                #input()

            res = np.column_stack(res)
            """
            off_one = copyme[:,:s]
            off_tow = other_copy[:,s:]
            res = np.append(off_one,off_tow, axis=1)
            """
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

                #res = np.append(p1,p2, axis=0)
                #print('before the the gene swap')
                #print(res)
                #return self.crossover(Seed(res))

            # other_copy is greater
            else:
                if far > r1:
                    p1 = copyme[:,:]
                    p2 = other_copy[r1:,:]
                else:
                    # i keep the slice even i skip genes in this case
                    p1 = copyme[:far,:]
                    p2 = other_copy[far:,:]

                #res = np.append(p1,p2, axis=0)
                #print('before the the gene swap')
                #print(res)
                #return other.crossover(Seed(res))

            res = np.append(p1,p2, axis=0)

        #print('the final result')
        #print(res)
        return Seed(res)
        

"""
Poplutaion class repsent the world of 2d plane that seed object can live 
inside it we used pygame libary for this world
"""
class Population:

    def __init__(self,path):
        pygame.init()
        self.best_one = None
        self.pop = []
        self.id = id(self)
        #corp the big image
        img = Image.open(path)
        if img.size[0] > 128 or img.size[1] > 128:
            name = f'{path}.jpg'
            img.resize((128,128)).save(name)
            #self.target is an image object
            self.target = pygame.image.load(name)
        else:
            self.target = pygame.image.load(path)

        #the main surface
        self.window = pygame.display.set_mode((256,360))
        self.window.fill((255,255,255))
        #draw the target image
        self.window.blit(self.target,(128,128))
        #self.ref load target image into 3d array 
        self.ref = pygame.surfarray.pixels3d(self.target)
        w,h,d = self.ref.shape
        self.size = (w,h)

        #self.alph = np.mean(pygame.surfarray.pixels_alpha(self.target_surf))
        #print(self.alph)
        #input()
        #the box that we draw circels on it not the main one

        self.window_p = pygame.Surface(self.size,pygame.SRCALPHA)
        self.window_p.fill((255,255,255))
        #number of Genretions
        self.i = 0
        self.w = 0


    def __str__(self,pAll=False):
        re = ''
        if pAll:
            for o in self.pop:
                re += f'\n{o.__str__()}'
        return f'ID: {self.id} Gen {self.i} best Memeber {self.best_one} \n {re}'


    def draw(self,org):
        # clear the prev Seed object from the canvas
        self.window_p.fill((255,255,255))
        # take a new clean canvas
        screan = self.window_p.copy()
        for gene in org.chrom:
            pos = (gene[5]*self.size[0] ,gene[6]*self.size[1])
            #pygame.color object that is simple
            #color = (int(gene[0]*255), int(gene[1]*255), int(gene[2]*255), int(gene[3]*64))

            c = ((gene[0]), (gene[1]), (gene[2]))
            color = tuple(map(lambda x: int(255 * x),  Color(hsl=c).rgb))
            color = color[:] + (int(gene[3]*64),)
            pygame.draw.circle(screan, color, pos,int((gene[4] * 32)))
            #print(pos)
            #print(gene)
            #print(color)
        #print(org.chrom)
        return screan 
        
    def culc_fitness(self,seed):

        crr_snapshot_surfce = self.draw(seed)
        crr = pygame.surfarray.pixels3d(crr_snapshot_surfce)
        array_dif = crr - self.ref
        #print(f'{crr[55,55]} {self.ref[55,55]} {array_dif[55,55]}')
        
        mean = np.mean(np.abs(array_dif)) #+ 1e-5 * seed.chrom.shape[0]
        #maxy = 255-self.ref
        #max_mean = np.mean(maxy)
        #print(mean,max_mean,mean/max_mean)
        seed.fitness = (1 - (mean/(255)))*100
        seed.pic_ref = crr_snapshot_surfce
           
    def mutate_best(self, seed, rate, scale, add, tries=10):
        """mutate and make the seed more diverse"""
        for i in range(choice(tries)):
            o = seed.mutate(rate=rate, scale=scale, add_chance=add)
            self.culc_fitness(o)
            if o.fitness > seed.fitness:
                return o
        return seed
    

    """
    create a population equals to 'pop_size' 
    etch seed object contains 'complexty' number of genes
    """
    def init(self,pop_size=50,complexty=10):
        for i in range(pop_size):
            crr = Seed(random((complexty,7)))
            self.culc_fitness(crr)
            self.pop.append(crr)
        #best fitness at index 0
        self.pop = sorted(self.pop, key=lambda x: -x.fitness)
        self.best_one = self.pop[0]
        self.i += 1



    def produse(self,mutation_rate=0.01,scale=0.1,add=0.1):
        new_pop = []
        #summ = sum([s.fitness for s in self.pop])
        w = 1 - np.linspace(0,0.2,len(self.pop))
        length = len(self.pop)
        #summ = sum(range(length))
        #w = [i / summ for i in range(length)]
        #w = [o.fitness / summ for o in self.pop]
        #print(w)
        #print(w/w.sum())
        self.w = w / w.sum()
        for i in range(length):
            a,b = choice(self.pop,2 , replace=False, p = self.w)
            new_seed = a.crossover(b)
            #print(a,b)

            self.culc_fitness(new_seed)
            """
            if random() < mutation_rate:
                         new_seed = new_seed.mutate(rate=mutation_rate)
            """
            new_seed = self.mutate_best(new_seed, mutation_rate ,scale, add)

            if new_seed.fitness > self.best_one.fitness:
                         self.best_one = new_seed
            #else:
                #cloning into new population
                #new_pop.append(self.best_one)

            new_pop.append(new_seed)

        #print(self.__str__(True))
        #print(self.w)
        #input()
        if abs(self.best_one.fitness - new_pop[-1].fitness) <= 0.0001:
            mutation_rate += abs(normal()* scale)
            scale = scale / mutation_rate 
            add += 0.001
            print(f'mutation rate has increced {mutation_rate}')
            #print(s)
            #input()
        
        slicee = choice(int(1+length*0.2))
        #print(slicee)
        top_fit_old = sorted(new_pop, key=lambda x: -x.fitness)[:slicee]

        top_fit_new = self.pop[:length-slicee] 
        self.pop = sorted(top_fit_new+top_fit_old, key=lambda x: -x.fitness)
        self.i += 1
        
            
    
"""
@desc: evovle one Population untel the Spcified Gen pramater
@Note: this function implement for any user wanan Play around 
with the pramters of the GA, u can comment the line 318 if u donot wanna 
see the evolotion prosses in detals but u still see the Drawing on the Surface
"""
def evolve(path,pop_size,Gen,rate=0.01):
    p = Population(path)
    p.init(pop_size)

    #print(p)
    for i in range(Gen):
        p.produse(mutation_rate = rate)
        p.window.blit(p.best_one.pic_ref ,(0,0))
        pygame.display.update()

        print(p.__str__(True))
        
"""
@descreption this method is created for testing performance of the GA 
the test are done in Performance.py file
@pram p is Population object to work with
@return list of best seeds object have lived so far
"""
def ev_with_p(p,pop_size,Gen,rate=0.01):
    p.init(pop_size)
    re = []
    for i in range(Gen):
        p.produse(mutation_rate = rate)
        re.append(p.best_one)
        p.window.blit(p.best_one.pic_ref ,(0,0))
        pygame.display.update()

        print(p.__str__(True))
    return re
# New
"""
def ev_with_p(p, pop_size, Gen, file_name_sample, start, rate=0.01):
    p.init(pop_size)
    re = []
    for i in range(Gen-1):
        p.produse(mutation_rate = rate)
        pas = time.time() - start
        re.append(p.best_one)
        p.window.blit(p.best_one.pic_ref ,(0,0))
        pygame.display.update()
        pygame.image.save(p.best_one.pic_ref, f'{file_name_sample}/Genration:{i+1} \
                _Time:{pas/60}_{p.best_one.fitness}.jpeg')

        print(p.__str__(True))
    return re
"""
# crossover Tests
"""
nrg = np.random.default_rng()
chrom = nrg.random((2,6))

p = Population('./Images/Mona_Lisa.jpg')
s = Seed(chrom)
o = Seed(nrg.random((2,6)))
re = s.crossover(o)

print('_________________________________')
print(s.chrom)
print('_________________________________')
print(o.chrom)
print('_________________________________')
print(re.chrom)
"""
# mutaion test
"""
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
#Start of The Main Program 
#evolve('./ll.jpeg', 15,50000,0.01)
