from __future__ import print_function
from __future__ import division
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Demo of reward for damaging mobs - create an arena filled with pigs and Zombie,
# and reward the agent positively for attacking Zombie, and negatively for attacking pigs.
# Using this reward signal to train the agent is left as an exercise for the reader...
# this demo just uses ObservationFromRay and ObservationFromNearbyEntities to determine
# when and where to attack.

from builtins import range
from past.utils import old_div
import MalmoPython
import os
import random
import sys
import time
import json
import random
import errno
import math
import malmoutils
import numpy as np

import agentMC

malmoutils.fix_print()

agent_host = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host)
recordingsDirectory = malmoutils.get_recordings_directory(agent_host)
video_requirements = '<VideoProducer><Width>860</Width><Height>480</Height></VideoProducer>' if agent_host.receivedArgument("record_video") else ''

# Task parameters:
MAX_DISTANCE = 40
MAX_ZOMBIES = 16
####### SPEED OF GAME #######
SPEED = 8
ARENA_WIDTH = MAX_DISTANCE
ARENA_BREADTH = MAX_DISTANCE

def getCorner(index,top,left,expand=0,y=0):
    ''' Return part of the XML string that defines the requested corner'''
    x = str(-(expand+old_div(ARENA_WIDTH,2))) if left else str(expand+old_div(ARENA_WIDTH,2))
    z = str(-(expand+old_div(ARENA_BREADTH,2))) if top else str(expand+old_div(ARENA_BREADTH,2))
    return 'x'+index+'="'+x+'" y'+index+'="' +str(y)+'" z'+index+'="'+z+'"'

def getSpawnEndTag(i):
    return ' type="mob_spawner" variant="' + ["Zombie", "Pig"][i % 2] + '"/>'

def getMissionXML(summary, numZoms=10, numCreeps=10):
    ''' Build an XML mission string.'''
    # We put the spawners inside an animation object, to move them out of range of the player after a short period of time.
    # Otherwise they will just keep spawning - as soon as the agent kills a Zombie, it will be replaced.
    # (Could use DrawEntity to create the pigs/Zombie, rather than using spawners... but this way is much more fun.)a
    return '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>''' + summary + '''</Summary>
        </About>

        <ModSettings>
            <MsPerTick>''' + str(int(50/SPEED)) + '''</MsPerTick>
        </ModSettings>
        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>14000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <AllowSpawning>false</AllowSpawning>
                <AllowedMobs>
                    Zombie
                    Skeleton
                </AllowedMobs>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,202*1,5*3,2;3;,biome_1" />
                <DrawingDecorator>
                    <DrawCuboid ''' + getCorner("1",True,True,expand=10,y=200) + " " + getCorner("2",False,False,y=246,expand=10) + ''' type="bedrock"/>
                    <DrawCuboid ''' + getCorner("1",True,True,y=201) + " " + getCorner("2",False,False,y=246) + ''' type="air"/>
                </DrawingDecorator>
               <DrawingDecorator>
               ''' + ('<DrawEntity type="Skeleton" x="0" y="207" z="13"/>\n' * numCreeps) + ('<DrawEntity type="Zombie" x="0" y="207" z="15"/>\n' * numZoms) + '''</DrawingDecorator>
               <ServerQuitWhenAnyAgentFinishes />
               <ServerQuitFromTimeUp timeLimitMs="120000"/>
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>The Hunter</Name>
            <AgentStart>
                <Placement x="0.5" y="201.0" z="0.5" pitch="20"/>
                <Inventory>
                    <InventoryItem type="diamond_axe" slot="0"/>
                    <InventoryItem type="diamond_helmet" slot="39"/>
                    <InventoryItem type="diamond_chestplate" slot="38"/>
                    <InventoryItem type="diamond_leggings" slot="37"/>
                    <InventoryItem type="diamond_boots" slot="36"/>
                </Inventory>
            </AgentStart>
            <AgentHandlers>
                <ContinuousMovementCommands turnSpeedDegs="1000"/>
                <MissionQuitCommands/>
                <ObservationFromRay/>
                <RewardForDamagingEntity>
                    <Mob type="Zombie" reward="1"/>
                    <Mob type="Skeleton" reward="1"/>
                </RewardForDamagingEntity>
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="'''+str(ARENA_WIDTH)+'''" yrange="2" zrange="'''+str(ARENA_BREADTH)+'''" />
                </ObservationFromNearbyEntities>
                <ObservationFromFullStats/>''' + video_requirements + '''
            </AgentHandlers>
        </AgentSection>

    </Mission>'''


validate = True
my_client_pool = MalmoPython.ClientPool()
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

episode_reward = 0
if agent_host.receivedArgument("test"):
    num_reps = 1
else:
    num_reps = 10000
fout = open('results.csv', 'w')
# Set up the agent
agent = agentMC.agentMC(agent_host, MAX_ZOMBIES, MAX_DISTANCE, 20)
for i in range(num_reps):
    print('episode:', i)
    for iRepeat in range(1, MAX_ZOMBIES):
        #########################################
        #       Set up the enviornment          #
        #########################################
        print('number of mobs:', iRepeat)
        numZoms = np.random.randint(0, iRepeat)
        mission_xml = getMissionXML("Go hunting! #" + str(iRepeat), numZoms, iRepeat - numZoms) 
        my_mission = MalmoPython.MissionSpec(mission_xml,validate)
        # Set up a recording
        my_mission_record = MalmoPython.MissionRecordSpec()
        if recordingsDirectory:
            my_mission_record.setDestination(recordingsDirectory + "//" + "Mission_" + str(iRepeat + 1) + ".tgz")
            my_mission_record.recordRewards()
            my_mission_record.recordObservations()
            my_mission_record.recordCommands(6)
            if agent_host.receivedArgument("record_video"):
                my_mission_record.recordMP4(24,2000000)

        max_retries = 3
        for retry in range(max_retries):
            try:
                # Attempt to start the mission:
                agent_host.startMission( my_mission, my_client_pool, my_mission_record, 0, "hunterExperiment" )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission",e)
                    print("Is the game running?")
                    exit(1)
                else:
                    time.sleep(2)

        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()

        time.sleep(1)
        #############################################
        #               main loop (my code)         #
        #############################################
        while world_state.is_mission_running:
            agent.last_reward = 0
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:

                # make agent look at mob
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                if not agent.lookAtMob(ob):
                    continue
                s = agent.state

                # epsilon greedy method
                a = agent.eGreedy(s)

                # take action a
                agent.act(a)

                if a == 'attack 1':
                    agent.lastAttackState = s
                # reset all possible actions to doing nothing
                agent_host.sendCommand('turn 0')
                time.sleep(.5/SPEED)
                agent_host.sendCommand('attack 0')
                agent_host.sendCommand('strafe 0')
                agent_host.sendCommand('move 0')

                # observe r, s'
                sPrime = agent.determinState(ob)
                if world_state.number_of_rewards_since_last_state > 0:
                    agent.last_reward += world_state.rewards[-1].getValue()
                r = agent.last_reward
                
                if agent.ZN == 0 | agent.lastZAH > agent.ZAH:
                    agent.addToState(agent.lastAttackState, a)
                else:
                    agent.addToState(s, a)
                # s<-s' done in the observation step
                #######################################
            agent.total_reward += agent.last_reward
            agent.last_reward = 0
        
        # tell the agent to learn after the episode (note that it is outside the while loop, so after episode ends)
        agent.learn()

        # output the total reward   
        print()
        print("round score:", agent.total_reward)
        episode_reward += agent.total_reward
        print()

        agent.lookAtMob(ob)

        if agent.ZAH > 1:
        # the agent is dead 
            break
        # mission has ended.
        for error in world_state.errors:
            print("Error:",error.text)
        if world_state.number_of_rewards_since_last_state > 0:
            # A reward signal has come in - see what it is:
            agent.total_reward += world_state.rewards[-1].getValue()

        agent.total_reward = 0
        time.sleep(1) # Give the mod a little time to prepare for the next mission.
    agent.total_reward = 0
    print("=" * 41)
    print('episode reward:', episode_reward)
    print("=" * 41)
    print('*' *55, '\n\n')
    fout.write(str(i) + ',' +str( episode_reward) + '\n')
    fout.flush()
    episode_reward = 0
    if agent.epsilon < .9:
        agent.epsilon += .01
fout.write(agent.V)
fout.close()