import time
import math
import numpy as np

actions = ["attack 1", 'strafe -1', 'strafe 1', 'move 1', 'move -1', 'wait']

class agentMC:
    
    def __init__(self, host, MAX_ZOMBIES, MAX_DISTANCE, MAX_HEALTH, ob=None):
        self.alpha = .01
        # consider shifting down to near 0
        self.epsilon = .1
        self.gamma = .9

        self.maxD = int( math.log(MAX_DISTANCE, 2)) + 1
        self.maxZN = MAX_ZOMBIES
        self.maxZAH = 20 #max average health of the zombies
        self.maxH = MAX_HEALTH
        
        self.state = (self.maxZN * self.maxD * self.maxH * self.maxZAH)
        self.state = int(self.maxD + 
                        (self.maxZN * self.maxD) + 
                        (self.maxZAH * self.maxZN * self.maxD) + 
                        (self.maxH * self.maxZAH * self.maxZN * self.maxD))
        
        self.D = int( math.log(MAX_DISTANCE, 2)) + 1
        self.ZN = MAX_ZOMBIES
        self.ZAH = 20 #max average health of the zombies
        self.H = MAX_HEALTH
        
        self.lastZN = 0
        self.lastZAH = 20 #max average health of the zombies
        self.lastH = MAX_HEALTH
        
        self.lastOb = ob
        self.ob = ob
        self.host = host
        
        self.lastAttackState = 0
        # TODO: consider removing these, as they are also initalized elsewhere
        self.total_reward = 0
        self.last_reward = 0
        self.zombie_population = 0
        self.zombie_distance = 0
        # initalixe with an invalid state
        self.lastAct = 'attack 0'
        self.x = 0
        self.z = 0
        self.current_yaw = 0
        # states by actions
        # I have learnied that this is the wrong way to do this
        # self.states = {'run away':0, 'kill':1, 'victory':2, 'dead':3}
        self.actions = ['attack 1', 'strafe 1', 'strafe -1', 'move -1', 'move 1']
        # states by actions, there is a seperate state for the distance and number of mobs
        # plus 2, state 0 for victory, state -1 for death
        self.V = np.random.random((self.state + 2, 5))
        self.KeepTrack = np.random.random((self.state + 2,5,2))
        print(self.state+2)
        print('inital Q:', self.V)
        self.rewards  = {'run away':0,
                            'kill':100, 
                            'victory':1000,
                            'dead':-1000,
                            'hurt':-10,
                            'damage':10
          }

    def eGreedy(self, s):
        e=self.epsilon
        V=self.V
        if np.random.rand() < e:
            a = np.argmax(V[s,:])
        else:
            a = np.random.randint(0, 5)
        return self.actions[a]

    def addToState(self, s, action):
        a = actions.index(action)
        # collecting the total reward fo the state, along with the totl visits
        self.KeepTrack[s][a][0] = self.last_reward
        self.KeepTrack[s][a][1] = 1

    def learn(self):
        for i in range(len(self.V)):
            for a in range(len(self.V[i])):
                if (self.KeepTrack[i][a][1] != 0):
                    print("Old V:", a, self.KeepTrack[i][a][0], self.V[i,a])
                    self.V[i,a] += self.alpha *((self.KeepTrack[i][a][0]/self.KeepTrack[i][a][1]) - self.V[i,a])
                    print("New V",a, self.KeepTrack[i][a][0], self.V[i,a])
                    self.KeepTrack[i][a][0] = 0
                    self.KeepTrack[i][a][1] = 0
    #s = self.state
    #sP = sPrime
    #aP = np.argmax(self.V[sP])
    #V(s)<-V(s)+a(G-V(s))
    #self.V[s,0] += self.alpha *('Gt' - )
    #self.Q[s,a] += self.alpha *(self.last_reward + (self.gamma * self.Q[sP, aP]) - self.Q[s,a])

    def lookAtMob(self, ob):
        if u'Yaw' in ob:
            self.current_yaw = ob[u'Yaw']
        if u'XPos' in ob:
            self.x = ob[u'XPos']
        if u'ZPos' in ob:
            self.z = ob[u'ZPos']
        # Use the nearby-entities observation to decide which way to move, and to keep track
        # of population sizes - allows us some measure of "progress".
        self.ZAH = 0
        if u'entities' in ob:
            entities = ob["entities"]
            num_Zombie = 0
            x_pull = 0
            z_pull = 0
            for e in entities:
                if e["name"] == "Zombie" or e["name"] == "Skeleton":
                    num_Zombie += 1
                    # Each Zombie contributes to the direction we should head in...
                    self.zombie_distance = max(0.0001, (e["x"] - self.x) * (e["x"] - self.x) + (e["z"] - self.z) * (e["z"] - self.z))
                    # Prioritise going after wounded Zombie. Max Zombie health is 8, according to Minecraft wiki...
                    weight = 21.0 - e["life"]
                    x_pull += weight * (e["x"] - self.x) / self.zombie_distance
                    z_pull += weight * (e["z"] - self.z) / self.zombie_distance
                    self.ZAH += e["life"]
                if e["name"] == "The Hunter":
                    self.H = e["life"]
            self.ZAH /= num_Zombie if num_Zombie > 0 else 1
            # Determine the direction we need to turn in order to head towards the "Zombieiest" point:
            yaw = -180 * math.atan2(x_pull, z_pull) / math.pi
            difference = yaw - self.current_yaw;
            while difference < -180:
                difference += 360;
            while difference > 180:
                difference -= 360;
            difference /= 180.0;
            self.host.sendCommand("turn " + str(difference))
            if num_Zombie != self.ZN:
                # Print an update of our "progress":
                self.ZN = num_Zombie
            if u'LineOfSight' in ob and num_Zombie > 0:
                los = ob[u'LineOfSight']
                return  los["type"]== "Zombie" or los["type"]== "Skeleton"
            else: 
                return True

        
    def determinState(self, ob):
        self.lastOb = self.ob
        self.ob = ob
        #print(self.ZN)
        if self.state == 0 and self.ZN == 0:
            print(ob)
            print('.', end='')
            return 0
        if self.lastZN > self.ZN:
            self.last_reward += self.rewards['kill']
        if self.lastZAH > self.ZAH:
            self.last_reward += self.rewards['damage']
        if self.lastH > self.H:
            self.last_reward += self.rewards['hurt']
        if self.ZN == 0:
            print('Victory!')
            self.last_reward += self.rewards['victory']/3
            self.state = 0
            self.host.sendCommand("quit")
        # if self.ZN > 0 and     
        # state = d + n*d where n = num zombies, and d = distance.
        if self.zombie_distance != 0:
            d = int(math.log(self.zombie_distance, 2)) if int(math.log(self.zombie_distance, 2)) < self.maxD else self.maxD
        else:
            d = 0
        n = self.ZN if self.ZN < self.maxZN else self.maxZN
        zah = int(self.ZAH)
        h = int(self.H)
        self.state = int(d + 
                        (n * self.maxD) + 
                        (zah * self.maxZN * self.maxD) + 
                        (h * self.maxZAH * self.maxZN * self.maxD))
        print(n, d, self.state, self.lastAct)
        self.lastZN = self.ZN


    def act(self, action='attack 1'):
        
        self.host.sendCommand(action)
        self.lastAct = action

        # Use the line-of-sight observation to determine when to hit and when not to hit:
        # we hard code that the agent is looking at the zombie
        '''if u'LineOfSight' in self.ob:
            los = self.ob[u'LineOfSight']
            type=los["type"]
            if type == "Zombie":
                self.host.sendCommand(action)
                self.lastAct = action
                #self.host.sendCommand("chat /replaceitem 'The Hunter' @p slot.weapon.offhand minecraft:stick 1" )'''
        return 0
