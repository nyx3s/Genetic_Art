from Imitation import * 

class Homogeneous:

    def __init__(self,path,num_pop=5,size=5):
        self.num_pop = num_pop
        self.size = size
        self.path = path
        self.avg_fit = None
        self.kingdom = [Population(path) for i in range(num_pop)]
        self.best_pop = None
        self.id = id(self)


    def __str__(self,pAll=False):
        re = ''
        if pAll:
            for p in self.kingdom:
                re += f'\n{p.__str__()}'
        return f'kingdom ID: {self.id} Pop_size in: {self.num_pop} best Population {self.best_pop} \n {re}'

    def paral_evolve(self,limit=5):
        for i in range(len(self.kingdom)):
            p = choice(self.kingdom)
            print(p,"is evovolving")
            evolve(p,self.path,self.size,limit, random())
            input()

    def rank_sort(self):
        return sorted(self.kingdom,key=lambda p:-p.best_one.fitness)

    def migrate(self):
        self.rank_sort()
        amount = choice(self.size)
        pop1,pop2 = choice(self.kingdom,2,p = w /w.sum())

        to_pop2 = pop1[:amount]
        to_pop1 = pop2[amount:]

        pop1.append(to_pop1)
        pop2.append(to_pop2)


"""
the start of the program still in test vase
"""
def evolve(p,path,pop_size,Gen,rate=0.01):
    p = Population(path)
    p.init(pop_size)

    #print(p)
    for i in range(Gen):
        p.produse(mutation_rate = rate)
        p.window.blit(p.best_one.pic_ref ,(0,0))
        pygame.display.update()

        print(p.__str__(True))
 
        



h = Homogeneous('Mona_Lisa.webp',5,10)

h.paral_evolve(5)
h.migrate()
print(h.best_pop.best_one)
