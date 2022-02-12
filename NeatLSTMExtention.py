import neat


# TODO: different chromosomes will have different expected input/output counts. that's not accounted for here
class LSTMGenome(DefaultGenome):
    """
    A genome for generalized neural networks.
    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.
    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

	chromosomes = []

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key

        # Custom setup
		nnLayers = [DefaultGenome(key), DefaultGenome(key)] # TODO: how many chromosomes will there be?

        # Fitness results.
        self.fitness = None

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""
		[chromo.configure_new(config) for chromo in self.chromosomes]

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        [chromo.configure_crossover(parentChromo1, parentChromo2, config) for chromo in self.chromosomes for (parentChromo1, parentChromo2) in zip(genome1.chromosomes, genome2.chromosomes)]

    def mutate(self, config):
        """ Mutates this genome. """
		[chromo.mutate(config) for chromo in self.chromosomes]

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

		# return the average distance between all chromosomes
		return sum(chromo.distance(otherChromo, config) for chromo, otherChromo in zip(self.chromosomes, other.chromosomes)) / len(self.chromosomes)

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """

		# return the average complexity of all chromosomes
        complexities = [chromo.size() for chromo in self.chromosomes] # [(Nnodes1, Nconn1), (Nnodes2, Nconn2), ...]
		complexitiesT = [[comp[j][i] for j in range(len(complexities))] for i in range(len(complexities[0]))] # [[Nnodes1, Nnodes2, ...], [Nconn1, Nconn2, ...]]

		averageComplexities = [sum(factor) / len(factor) for factor in complexitiesT]
		return averageComplexities

    def __str__(self):
        s = "Key: {0}\nFitness: {1}\nChromosomes:\n".format(self.key, self.fitness)
        chromosomeStrings = ['\t' + chromo.__str__().replace('\n', '\n\t') for chromo in self.chromosomes]
		numberedChromosomeStrings = ['('+i+')'+st for st, i in zip(chromosomeStrings, range(len(chromosomeStrings)))]

		return s + '\n'.join(numberedChromosomeStrings)



from neat.nn import FeedForwardNetwork

CELL_STATE_LENGTH = 10
class LSTM(object):

    def __init__(self, inputs, outputs, node_evals):
        # self.input_nodes = inputs
        # self.output_nodes = outputs
        # self.node_evals = node_evals
        # self.values = dict((key, 0.0) for key in inputs + outputs)

		self.nn_layers = [] # array of FeedForwardNetwork objects
		self.cell_state = [0]*CELL_STATE_LENGTH # TODO: CELL_STATE_LENGTH should be a parameter passed that gets read from the config
		self.previous_output = [0]*len(outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        # for k, v in zip(self.input_nodes, inputs):
        #     self.values[k] = v

        # for node, act_func, agg_func, bias, response, links in self.node_evals:
        #     node_inputs = []
        #     for i, w in links:
        #         node_inputs.append(self.values[i] * w)
        #     s = agg_func(node_inputs)
        #     self.values[node] = act_func(bias + response * s)

        # return [self.values[i] for i in self.output_nodes]

		# TODO: this

		def forget_gate(cell_state):
			gate_output = self.nn_layers[0].activate(self.previous_output + inputs)
			return [cell * gate_out for cell, gate_out in zip(cell_state, gate_output)]

		def replace_new_gate(cell_state):
			replace_gate_output = self.nn_layers[1].activate(self.previous_output + inputs)
			new_gate_output = self.nn_layers[2].activate(self.previous_output + inputs)

			return [cell + rep*new for cell, rep, new in zip(cell_state, replace_gate_output, new_gate_output)]

		def output_gate(cell_state):
			hide_state_output = self.nn_layers[3].activate(self.previous_output + inputs)
			transform_state_output = self.nn_layers[4].activate(cell_state)

			return [hide*trans for hide, trans in zip(hide_state_output, transform_state_output)]


		self.cell_state = forget_gate(self.cell_state)
		self.cell_state = replace_new_gate(self.cell_state)

		return output_gate(self.cell_state)
		

		



    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (an LSTM). """

		# TODO: different chromosomes will have different expected input/output counts. that's not accounted for here
		nn_layers = [FeedForwardNetwork.create(chromo, config) for chromo in genome.chromosomes]
        return LSTM(config.genome_config.input_keys, config.genome_config.output_keys, nn_layers)









# not technically part of an LSTM, but required for my implementation
class MemoryDB:
    allMemories = list()
    memoryLookup = list() # of form: [idIndex: { idValue1: {memories}, idValue2: {memories}, ... }, ...]

    idVectorLen = None
    maxNumMemories = 100

    numMemoriesReturnedByLookup = 5

    def __init__(self, idVectorLen = 10):
        self.idVectorLen = idVectorLen

        for i in range(idVectorLen):
            self.memoryLookup.append(dict())


    def addMemory(memory, idVector):
        self.allMemories.append((memory, idVector))

        for i, tag in zip(range(self.idVectorLen), idVector):
            if memoryLookup[i][tag] == None:
                memoryLookup[i][tag] = set()
            memoryLookup[i][tag].add(memory)

        # if we have too many memories, remove them
        if len(self.allMemories) > self.maxNumMemories:
            (memToRemove, idVecToRemove) = self.allMemories.pop(0)
            
            for i, tag in zip(range(self.idVectorLen), idVecToRemove):
                memoryLookup[i][tag].remove(memToRemove)

    def lookup(idVector):
        memoriesByScore = {} # {score: [memory]}

        retval = []
        memory_groups = [
                self.memoryLookup[i][idVector[i]].copy() 
                if self.memoryLookup[i][idVector[i]] != None 
                else set() 
            for i in range(self.idVectorLen)
        ] # all memories that have at least one matching id tag, grouped by which tag matches

        idealScore = self.idVectorLen
        settle_for_less = 0

        while not all(len(group) == 0 for group in memory_groups): # while we haven't searched all memories
            
            smallest_group = smallestCollection(memory_groups, lambda group: len(group) <= 0)

            for memory in smallest_group:
                score = 0
                for group in memory_groups:
                    if group[memory]:
                        score += 1
                        group.remove(memory)

                if score == idealScore-settle_for_less:
                    retval.append(memory)
                else:
                    if memoriesByScore[score] == None:
                        memoriesByScore[score] = []
                    memoriesByScore[score].append(memory)

                if len(retval) >= self.numMemoriesReturnedByLookup:
                    return retval[0:self.numMemoriesReturnedByLookup]

            #  if we didn't find enough memories with the current best possible score, supplement retval with memories of the next best possible score
            retval += memoriesByScore[idealScore-settle_for_less-1]
            memoriesByScore[idealScore-settle_for_less-1] = []
            if len(retval) >= NUM_MEMORIES_RETURNED:
                return retval[0:self.numMemoriesReturnedByLookup]

            settle_for_less += 1

        #  there just straight up aren't enough memories of score 1 or greater to fill an array of len NUM_MEMORIES_RETURNED
        return retval
	

def smallestCollection(collections, ignoreCondition = lambda collection: False):
    smallestLen = None
    smallest = None

    for collection in collections:
        if ignoreCondition(collection):
            continue

        l = len(collection)
        if smallestLen == None or l < smallestLen:
            smallestLen = l
            smallest = collection
    
    return smallest

    
        