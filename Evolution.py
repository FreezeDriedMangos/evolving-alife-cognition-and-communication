"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat
import visualize
import Agent
import World

grandparents = []
parents = []
cur_gen = []

p = None
world = None

def eval_genomes(genomes, config):
    # init
    global grandparents
    global parents
    global cur_gen

    grandparents = parents
    parents = cur_gen
    cur_gen = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        agent = Agent.Agent(net, *p.reproduction.ancestors[genome_id])

        cur_gen.append((genome_id, genome, agent))

    # simulate
    world.init_agents([agent for (genome_id, genome, agent) in grandparents+parents+cur_gen]) # skips agents that are dead
    for i in range(10000): # simulate 10,000 'seconds'
        world.iteration()

    # rate performance
    for ((genome_id, genome, agent), index) in zip(cur_gen, range(len(cur_gen))):
        # most food eaten, least damage taken gets max fitness
        # genome.fitness = 0 if agent.max_age <= agent.age else agent.max_age
        genome.fitness = agent.max_age - agent.age

        # TODO: try (num_agents_in_species_still_alive / num_agents_in_species)
        # TODO: try (agent.max_age - agent.age) * (num_agents_in_species_still_alive / num_agents_in_species)


def run(config_file):
    # Load configuration.
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
    print("Num inputs: " + str(Agent.TOTAL_INPUT_COUNT))
    print("Num outputs: " + str(Agent.TOTAL_OUTPUT_COUNT))
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)