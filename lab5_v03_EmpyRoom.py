#%% libraries
import numpy as np 
import random
import math 
import heapq
import matplotlib  
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#%% README
# random cheomosome generator with length of 54 
# fittness function -> provided
# for 50 chromosome:
    # calculate fittness
    # pick the elite 
    # crossover function -> written: ok
    # mutation function -> written: TODO: make it possible to keep the same gene : DONE
        # reason: not to lose a valuable gene
# make 200 generation : each generation have 50 poplulation(chromosomes)
    # perform fittness calculation before and after crossover & mutation on each generation
    # next_generation = prev_generation 
# creat a room with furniture

#%% painter_play/fitness func
# %Floor painter algorithm
# %Code translated into python Fiona Skerman from matlab code written by Alex Szorkovszky for UU Modelling Complex Systems

# %INPUTS
# %rules: 54-cell array with one of three actions: 0(no turn) 1(turn left) 2(turn right) 3(random turn left/right)
# %room: MxN matrix defining a rectangular room with each square either 0(empty) or 1(furniture) or 2(painted)


# Chromosome (54-cell rule array) encodes action for each of 54 different scenarios
# no turn, turn left, turn right, turn left/right with 50/50 
# let [c, f, l, r] denote the states of current, forward, left and right squares, 
# then rule for that position is at position i=2(9f+3l+r)+Indicator[c=2] in chromosone.


#Painter has a position x,y in MxN matrix, and a direction -1 Left, 0 Up, 1 Right, -2 Down.

# Each time step consists of three parts
# a) according to rule on current environment update direction either 0 no turn, 1 turn left, 2 turn right, 3 random turn left/right
# b) if on unpainted square, paint it
# c) go forwards if possible

# OUTPUTS
# score: percentage of empty space painted
# xpos: x positions over time
# ypos: y positions over time

def painter_play(rules,room):
  #returns score, xpos, ypos

  M, N = room.shape

  #Calculates number of squares t to be painted. / #steps allowed
  t=M*N - room.sum()
  t=int(t)
  # print('t value: ',t) #Raam

  # add walls
  # env 0 - empty square, 1 - wall/obstruction, 2 - painted square
  env = np.ones((M+2,N+2));
  for i in range(1, M+1):
    for j in range(1, N+1):
      env[i][j]=0

  #new room size including walls
  M=M+2
  N=N+2

  xpos=[np.nan]*(t+1)
  ypos=[np.nan]*(t+1)

  # %random initial location   
  while True:
    xpos[0]=math.floor(M*random.random())
    ypos[0]=math.floor(N*random.random())
    if env[xpos[0], ypos[0]] == 0:
      break


  # random itial orientation (up=0,left=-1,right=+1,down=-2)
  direction = math.floor(4*random.random()) - 2

  # initial score
  score = 0

  for i in range(t):
    # directions -1: Left, 0: Up, 1: Right, 2: Down
    # dx, dy of a forward step (given current direction)        
    dx = divmod(direction,2)[1]
    if direction == -1:
      dx = -1 * dx

    dy = divmod(direction+1,2)[1]
    if direction == -2: 
      dy = -1*dy




    # dx, dy of a square to right (given currection direction)  
    r_direction=direction+1
    if r_direction == 2:
      r_direction = -2

    dxr = divmod(r_direction,2)[1]
    if r_direction == -1:
      dxr = -1 * dxr
    dyr = divmod(r_direction+1,2)[1]
    if r_direction == -2: 
      dyr = -1*dyr

    # evaluate surroundings (forward,left,right)
    local = [env[xpos[i] + dx, ypos[i] + dy], env[xpos[i] - dxr, ypos[i] - dyr], env[xpos[i] + dxr, ypos[i] + dyr]]      
      
    #localnum= 2* np.dot([9,3,1], local) if env[xpos[i], ypos[i]] == 2 else 2* np.dot([9,3,1], local) + 1
    localnum= int(2* np.dot([9,3,1], local))
    if env[xpos[i], ypos[i]] == 2:
       localnum += 1
     
    #use turning rule 1 'turn left', 2 'turn right', 3 'turn left/right 50/50 probabilities'
    if rules[localnum] == 3:
      dirchange = math.floor(random.random()*2)+1
    else:
      dirchange = rules[localnum]

    if dirchange == 1:
      direction = direction - 1
      if direction == -3:
        direction = 1
    elif dirchange == 2:
      direction = direction + 1
      if direction == 2:
        direction = -2

    dx = divmod(direction,2)[1]
    if direction == -1:
      dx = -1 * dx

    dy = divmod(direction+1,2)[1]
    if direction == -2: 
      dy = -1*dy  

    # paint square
    if env[xpos[i],ypos[i]]==0:
      env[xpos[i],ypos[i]] = 2
      score = score + 1
      
    # go forward if possible - stay put if wall/obstacle ahead
    if env[xpos[i]+dx, ypos[i]+dy] == 1:
      xpos[i+1] = xpos[i]
      ypos[i+1] = ypos[i]
    else:
      xpos[i+1] = xpos[i]+dx
      ypos[i+1] = ypos[i]+dy      
  

  # %normalise score by time            
  score = score/t  

  return score, xpos, ypos, env
# tests
test_room=np.zeros((8,7)) # Test room here is an empty room
test_rules=np.ones((54,1))

for i in range(len(test_rules)):
  test_rules[i]=3

fitness, xi , yi, env = painter_play(test_rules, test_room)[0],painter_play(test_rules, test_room)[1],painter_play(test_rules, test_room)[2],painter_play(test_rules, test_room)[3]
test_painter = painter_play(test_rules, test_room)
print('test info\n fitness:',test_painter[0])
print('x pos  :',test_painter[1])
print('y pos  :',test_painter[2])
print('env    :',test_painter[3])

# %% TEST animation TRAJECTORY

X = test_painter[1]  #np.linspace(0, 2*np.pi, 100)
Y = test_painter[2] #np.sin(X)

fig, ax = plt.subplots(1,1)
ax.set_xlim([0, 9])
ax.set_ylim([0, 9])
# cursor(red square)
graph, = ax.plot([], [])
dot, = ax.plot([], [], 's', color='red')

yrange = np.arange(0,9,1)
plt.yticks(yrange)
xrange = np.arange(0,9,1)
plt.xticks(xrange)

data = np.ones((7,8))
#convert the data array into x-y coordinates
#assume (0,0) is the lower left corner of the array
def array_to_coords(data):
    for y, line in enumerate(reversed(data)):
        for x, point in enumerate(line):
            if point == 1:
                yield(x, y)
points = list(array_to_coords(data))

x,y = zip(*points)
plt.plot(y, x, 'o')

def myFunc(i):
    graph.set_data(X[:i],Y[:i])
    dot.set_data(X[i],Y[i])

anim = animation.FuncAnimation(fig, myFunc, frames=len(X), interval=200)
# Set ggplot styles and update Matplotlib with them.
ggplot_styles = {
    'axes.edgecolor': 'white',
    'axes.facecolor': 'EBEBEB',
    'axes.grid': True,
    'axes.grid.which': 'both',
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
    'grid.color': 'white',
    'grid.linewidth': '1.2',
    'xtick.color': '555555',
    'xtick.major.bottom': True,
    'xtick.minor.bottom': False,
    'ytick.color': '555555',
    'ytick.major.left': True,
    'ytick.minor.left': False,
}

plt.rcParams.update(ggplot_styles)

anim.save(r'testRoom.mp4')
plt.show()
#%% TEST TRAJECTROY PLOT
# plt.figure(figsize=(8, 7, dpi=300))
plt.plot(X, Y, '-', color='g', linewidth=10)

# plt.plot(yi, xi, 'd') #, linewidth=100)
# plt.plot(yi, xi, 's', linewidth=100)

default_y_ticks = range(0,9,1)
plt.yticks(default_y_ticks)

# plt.yticks(default_y_ticks, yi)
default_x_ticks = range(0,8,1)
# plt.xticks(default_x_ticks, xi)
plt.xticks(default_x_ticks)

plt.show() 
#%% fitness calculation Func
# DONE
'''
fit_ave(chromosomes)
    Clculate average fitness
    Arguments:
        chromosomes: a list of 50 chromosomes(50x54) which belong to a generation
    Returns:
        The mean value of fitness of that generation
        List of fitness of each chromosome
        The index of the best chromosome of a generation 
'''
def fit_ave(chromosomes):
    for i in range(0, 50):
        fitness = np.zeros(50)
        fitness[i] = painter_play(chromosomes[i],room)[0]
    return np.mean(fitness) , fitness, np.argmax(fitness)#,max(fitness)

#%% Elite Picker Func
'''
elite(chromosomes)
    Clculate average fitness
    Arguments:
        chromosomes: a list of 50 chromosomes which belong to a generation
    Returns:
        List of Top 10 member(chromosomes) of that generation
'''
def elite(chromosomes, fitness):
    elite_ind = heapq.nlargest(10, range(len(fitness)), fitness.take)
    elite_list = chromosomes[elite_ind]
    return elite_list

#%% # crossover function #  OK
'''
crossover(par1, par2)
    make 2 babe
    Arguments:
        par1: chromosomes of length54
        par2: chromosomes of length54
    Returns:
        babe(chromosomes of length54)
'''
def crossover(par1, par2):
    spot = random.randint(0, 53)
    babe1 = np.append(par1[:spot], par2[spot:])
    #babe2 = np.append(par2[:spot], par1[spot:])
    return babe1#, babe2

# testBabe1, testBabe2 = crossover(par1, par2)
# print("fitness after crossover babe1",painter_play(testBabe1, room)[0])
# print("fitness after crossover babe2",painter_play(testBabe2, room)[0])
#%% mutation Func
'''
f = np.zeros((200))
for i in range(200):
    w = crossover(elite_list[0], elite_list[1])
    f[i] = painter_play( w, room)[0]
'''    
# mutation  
# print('testBabe1 befor mutation',testBabe1)
def mutation(string):
    # pick mutation spot 
    spot = random.randint(0, 53)
    mutation_rate = random.randint(0,1) 
    string[spot] = ((mutation_rate*string[spot]+1)%4) #TOCHECK: modular
    # print('mutation spot:',spot,'mutation rate', mutation_rate, 'string',string[spot])
    return string

# newbabe = np.zeros(54)
# testBabe1 = mutation(testBabe1)
# print('testBabe1 after mutation',testBabe1)

# print('fitness after mutation babe1', painter_play(testBabe1, room)[0])

#%% # generating G = 200 generation from 1st generation(50 chromosoms)

print("Main begins ...")

# 20x40 empty room
room = np.zeros((20,40))
# 50 chromosomes, length of each 54, 200 generation
# each chromosome consist of 54 genes 
# each gene belong to {0,1,2,3}
G = 200 # Generation#
chromosomes = np.zeros((50,54)) #list of chromosomes of one generation
fitness_ave = np.zeros((1,G)) # keep the record of fitness of each chromosome in a generation
generation  = np.zeros((50,54,G)) # keep the record of all generations chromosomes

for i in range(chromosomes.shape[0]): 
    for j in range(chromosomes.shape[1]): 
        chromosomes[i][j] = random.randint(0, 3)
        generation[0][i][j] = chromosomes[i][j] #OK

fisrt_generation =  chromosomes
print("Fisrt genration", fisrt_generation) 


for j in range(G):
    # Calculating fitness list & everage fitness of first generation
    fit_mean, fitness, _ = fit_ave(chromosomes)
    fitness_ave[0,j] = fit_mean
    # print('fit_mean:', fit_mean)
    # print('\nfitness', fitness)
    # picking elites
    elite_list = elite(chromosomes, fitness)
    # print("Elites list", elite_list)
    next_generation = np.zeros((50,54))
    # crossover 
    for i in range(50):
        s, ss = random.randint(0,9), random.randint(0,9) # 9  due to the size of our selected elite
        next_generation[i,:] = crossover(elite_list[s,:], elite_list[ss,:])
        # mutation 
        next_generation[i,:] = mutation(next_generation[i,:])
    chromosomes = next_generation
    generation[:,:,j] = chromosomes
    
#%% EXTRACTING DATA FOR TRAJECTORY OF BEST CHROMOSOME
best_generation_ind = np.argmax(fitness_ave) 
print("maximum fitness average value", np.max(fitness_ave))
print("\nfrom generation", best_generation_ind)
print( generation[:,:,np.argmax(fitness_ave)])  
best_gener = generation[:,:,best_generation_ind]

best_chrom = best_gener[fit_ave(best_gener)[2]]
# xi, yi = painter_play(best_chrom, room)[1], painter_play(best_chrom, room)[2]
# xi = np.zeros((50))
painted = painter_play(best_chrom, room)
xi= painted[1]
yi= painted[2]
print('best chrom fitt value:', painted[0])
#%% plot final set of chromosomes (chromosomes of the 200th generation)
colors = 'w y g r'.split()
cmap = matplotlib.colors.ListedColormap(colors, name='colors', N=None)
plt.title("Chromosomes of the last generation")
plt.imshow(next_generation, cmap=cmap)
plt.savefig("Chromosomes of the last generation.png", dpi=300)
plt.show()
#%% Plot the average fitness over 200 generations
generation = np.linspace(1, G, G)
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(generation, fitness_ave.reshape(G,1),'g-s', label = "Average fitness")
# plt.plot(generation, fitness_ave.reshape(G,1),'g', '-s')

plt.xlabel('Generation')
plt.ylabel('The average fitness')
plt.title('The average fitness for %d generations' %G) 
# plt.legend(loc="upper left")
plt.legend()
plt.savefig("The average fitness of the population %d"%G)
plt.show()      

#%%# plot trajectory 
plt.plot(yi, xi, '-')
default_y_ticks = range(0,21,1)
# plt.yticks(default_y_ticks, xi)
plt.title('TRAJECTORY OF BEST CHROMOSOME')
plt.savefig("TRAJECTORY OF BEST CHROMOSOME Over population %d"%G)
plt.show()

#%%# animate trajectory empty room

X = yi
Y = xi 

fig, ax = plt.subplots(1,1)
ax.set_xlim([0, 41])
ax.set_ylim([0, 21])
# cursor(red square)
graph, = ax.plot([], [])
dot, = ax.plot([], [], 's', color='red')

yrange = np.arange(0,21,1)
plt.yticks(yrange)
xrange = np.arange(0,41,1)
plt.xticks(xrange)

'''
data = np.ones((40,20))
#convert the data array into x-y coordinates
#assume (0,0) is the lower left corner of the array
def array_to_coords(data):
    for y, line in enumerate(reversed(data)):
        for x, point in enumerate(line):
            if point == 1:
                yield(x, y)
points = list(array_to_coords(data))

x,y = zip(*points)
plt.plot(y, x, 'o')
'''

def myFunc(i):
    graph.set_data(X[:i],Y[:i])
    dot.set_data(X[i],Y[i])

anim = animation.FuncAnimation(fig, myFunc, frames=len(X), interval=200)
# Set ggplot styles and update Matplotlib with them.
ggplot_styles = {
    'axes.edgecolor': 'white',
    'axes.facecolor': 'EBEBEB',
    'axes.grid': True,
    'axes.grid.which': 'both',
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
    'grid.color': 'white',
    'grid.linewidth': '1.2',
    'xtick.color': '555555',
    'xtick.major.bottom': True,
    'xtick.minor.bottom': False,
    'ytick.color': '555555',
    'ytick.major.left': True,
    'ytick.minor.left': False,
}

plt.rcParams.update(ggplot_styles)

anim.save(r'EmptyRoom.mp4')
plt.show()

#%%
