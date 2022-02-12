
from __future__ import print_function
import os
import neat
import visualize
import Agent
import World
import GUI

WALL_CHANGE_SPEED = 0.5
GEN_TIMER_CHANGE_SPEED = 0.25

grandparents = []
parents = []
cur_gen = []
generation_number = 0

p = None
world = None


# TODO: collision detection
# TODO: raycasting (sight)
# TODO: food (that adds to max_age)
# TODO: a gui to show what's happening (toggleable to run faster/slower)
# TODO: seed the RNG

# Food location randomization:
# use the random walk algorithm from AI for games (have a walk direction and randomly add/subtract a small amount from it)
# grow the distance walked based on the generation number
# if the walk direction points outside of the world area, reflect the portion of the vector that's outside the bounds off the wall
# 
# eventual improvement: instead of directly using the NEAT nn, use it to generate 4 other NNs
#   - a convolutional one to process vision
#   - (recurrent) one to take the convolutional output and sensory info as input and output "state" (an N length vector)
#   - (recurrent) one to take state from the previous frame and output reward (a single float from -1 to 1)
#   - (recurrent) one to take state as input and output action - this one will also be tuned using RL
#          (should this one also handle graph operations, should I have another precedesor to handle graph ops, or should I just toss graph ops?)


def eval_genomes(genomes, config):
    # init
    global grandparents
    global parents
    global cur_gen
    global generation_number

    grandparents = parents
    parents = cur_gen
    cur_gen = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config) # TODO: in order to generate LSTMs instead, just do LSTM.create(genome, config)
        agent = Agent.Agent(net, *[parent[2] for parent in parents if parent[0] in p.reproduction.ancestors[genome_id]])

        cur_gen.append((genome_id, genome, agent))

    world.update_walls(int((generation_number*WALL_CHANGE_SPEED)**2))

    # simulate
    world.init_agents([agent for (genome_id, genome, agent) in grandparents+parents+cur_gen]) # skips agents that are dead
    for i in range(min(10000, 100 + int((generation_number*GEN_TIMER_CHANGE_SPEED)**2))): # simulate 10,000 'seconds'
        world.iteration()
        GUI.draw(world)

    # rate performance
    for ((genome_id, genome, agent), index) in zip(cur_gen, range(len(cur_gen))):
        # most food eaten, least damage taken gets max fitness
        # genome.fitness = 0 if agent.max_age <= agent.age else agent.max_age
        genome.fitness = agent.max_age - agent.age

        # TODO: try (num_agents_in_species_still_alive / num_agents_in_species)
        # TODO: try (agent.max_age - agent.age) * (num_agents_in_species_still_alive / num_agents_in_species)

    generation_number += 1


def run(config_file):
    # Load configuration.
    # TODO: to enable LSTM evolution, replace neat.DefaultGenome with LSTMGenome
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    global world
    world = World.World()

    global p
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    GUI.init()

    print("Num inputs: " + str(Agent.TOTAL_INPUT_COUNT))
    print("Num outputs: " + str(Agent.TOTAL_OUTPUT_COUNT))
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)