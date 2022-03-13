import neat
import math


# HARDCODE EVERYTHING YEEEEHAAAW
CELL_STATE_LENGTH = 7
ID_VECTOR_LENGTH = 4


class LSTMGenome(neat.DefaultGenome):
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
        param_dict['node_gene_type'] = neat.genes.DefaultNodeGene
        param_dict['connection_gene_type'] = neat.genes.DefaultConnectionGene
        return neat.genome.DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key

        # Custom setup
        self.chromosomes = [
            neat.DefaultGenome(key), 
            neat.DefaultGenome(key), 
            neat.DefaultGenome(key), 
            neat.DefaultGenome(key), 
            neat.DefaultGenome(key), 
            neat.DefaultGenome(key) # this one only used for LSTM_WithMemory
        ] # TODO: how many chromosomes will there be?

        # Fitness results.
        self.fitness = None

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""
        [chromo.configure_new(config) for chromo in self.chromosomes]

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        [chromo.configure_crossover(parentChromo1, parentChromo2, config) for chromo in self.chromosomes for (parentChromo1, parentChromo2) in zip(genome1.chromosomes, genome2.chromosomes)]
        
        if hasattr(genome1, 'memories') and hasattr(genome2, 'memories'):
            self.memories = MemoryDB.create(genome1.memories, genome2.memories)

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
        complexitiesT = [[complexities[j][i] for j in range(len(complexities))] for i in range(len(complexities[0]))] # [[Nnodes1, Nnodes2, ...], [Nconn1, Nconn2, ...]]

        averageComplexities = [sum(factor) / len(factor) for factor in complexitiesT]
        return averageComplexities

    def __str__(self):
        s = "Key: {0}\nFitness: {1}\nChromosomes:\n".format(self.key, self.fitness)
        chromosomeStrings = ['\t' + chromo.__str__().replace('\n', '\n\t') for chromo in self.chromosomes]
        numberedChromosomeStrings = ['('+i+')'+st for st, i in zip(chromosomeStrings, range(len(chromosomeStrings)))]

        return s + '\n'.join(numberedChromosomeStrings)



from neat.nn import FeedForwardNetwork

class LSTM(object):

    def __init__(self, inputs, outputs, nn_layers, cell_state_len=CELL_STATE_LENGTH):
        # self.input_nodes = inputs
        # self.output_nodes = outputs
        # self.node_evals = node_evals
        # self.values = dict((key, 0.0) for key in inputs + outputs)

        self.cell_state_len = cell_state_len
        self.nn_layers = nn_layers # array of FeedForwardNetwork objects
        self.cell_state = [0]*self.cell_state_len # TODO: cell_state_len should be a parameter passed that gets read from the config
        # self.previous_output = [0]*len(outputs)

        self.num_inputs = len(inputs)
        self.num_outputs = len(outputs)
        self.last_output = [0]*self.num_outputs

    def activate(self, inputs):
        if self.num_inputs != len(inputs):
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

        inputs = inputs + self.last_output

        def forget_gate(cell_state):
            gate_output = self.nn_layers[0].activate(inputs)     # sigmoid
            return [cell * gate_out for cell, gate_out in zip(cell_state, gate_output)]

        def replace_new_gate(cell_state):
            replace_gate_output = self.nn_layers[1].activate(inputs) # sigmoid
            new_gate_output = self.nn_layers[2].activate(inputs)     # tanh

            return [cell + rep*new for cell, rep, new in zip(cell_state, replace_gate_output, new_gate_output)]

        def output_gate(cell_state):
            hide_state_output = self.nn_layers[3].activate(inputs)   # sigmoid
            transform_state_output = self.nn_layers[4].activate(cell_state)                 # tanh

            return [hide*trans for hide, trans in zip(hide_state_output, transform_state_output)]


        self.cell_state = forget_gate(self.cell_state)
        self.cell_state = replace_new_gate(self.cell_state)

        self.last_output = output_gate(self.cell_state)
        return self.last_output
        

        



    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (an LSTM). """

        # HARDCODE EVERYTHING YEEEEHAAAW

        num_top_level_inputs = config.num_inputs # the number of inputs the LSTM appears to have to outside observers
        num_top_level_outputs = config.num_outputs # the number of outputs the LSTM appears to have to outside observers
        internal_num_inputs = num_top_level_inputs + num_top_level_outputs # the number of inputs after secretly concatenating extra values to the outsiders' given input

        layer_details = [
            ('sigmoid', internal_num_inputs, CELL_STATE_LENGTH), # cell state managers
            ('sigmoid', internal_num_inputs, CELL_STATE_LENGTH),
            ('tanh', internal_num_inputs, CELL_STATE_LENGTH),

            ('sigmoid', internal_num_inputs, num_top_level_outputs), # outputters
            ('tanh', CELL_STATE_LENGTH, num_top_level_outputs)
        ]

        return LSTM.create_given_layer_details(genome, config, layer_details)


    @staticmethod
    def create_given_layer_details(genome, config, layer_details, constructor=None):
        """ Receives a genome and returns its phenotype (an LSTM). """
        if constructor == None:
            constructor = LSTM

        nn_layers = []

        old__config__genome_config__activation_defs__get = config.genome_config.activation_defs.get
        old__config__input_keys = config.genome_config.input_keys if hasattr(config.genome_config, 'input_keys') else None
        old__config__output_keys = config.genome_config.output_keys if hasattr(config.genome_config, 'output_keys') else None
        old__config__num_inputs = config.num_inputs if hasattr(config, 'num_inputs') else None
        old__config__num_outputs = config.num_outputs if hasattr(config, 'num_outputs') else None
        for chromo, (activation, num_inputs, num_outputs) in zip(genome.chromosomes, layer_details):
            # I am very not happy with this, but it gets the job done
            config.genome_config.activation_defs.get = lambda _: old__config__genome_config__activation_defs__get(activation)
            
            config.genome_config.input_keys = [-i - 1 for i in range(num_inputs)] # ripped straight out of https://github.com/CodeReclaimers/neat-python/blob/c2b79c88667a1798bfe33c00dd8e251ef8be41fa/neat/genome.py
            config.genome_config.output_keys = [i for i in range(num_outputs)]

            config.num_inputs = num_inputs
            config.num_outputs = num_outputs

            layer = FeedForwardNetwork.create(chromo, config)
            nn_layers.append(layer)

        config.genome_config.activation_defs.get = old__config__genome_config__activation_defs__get
        config.genome_config.input_keys = old__config__input_keys    # return config.input_keys/output_keys to the values an outside observer to the LSTM would expect
        config.genome_config.output_keys = old__config__output_keys  
        config.num_inputs = old__config__num_inputs
        config.num_outputs = old__config__num_outputs       

        return constructor(config.genome_config.input_keys, config.genome_config.output_keys, nn_layers)







class LSTM_WithMemory(LSTM):
    
    MEMORY_IMPORTANCE_THRESHOLD = 0.5
    ID_TAG_RANGE = 256
    # IMPORTANCE_SCALING = 10

    def __init__(self, *args,**kwargs):
        LSTM.__init__(self, *args,**kwargs)
        self.memories = MemoryDB(idVectorLen=ID_VECTOR_LENGTH)


    # STEPS:
    # 1. Get the id vector and importance the current cell state would have if it were to become a memory
    # 2. Look up memories relevant to that id vector
    # 3. If the current cell state's importance (given by the first element of its id vector) is high enough, store it for future lookups
    # 4. process the relevant memories into a context vector by doing a weighted sum according to thier relative importances ( context_vector = vector_sum(memory.memory * (memory.importance / total_importance) for memory in lookup_results) )
    # 5. feed this into the main LSTM along with the regular inputs
    def activate(self, inputs):
        # 1
        memory_identifier = self.nn_layers[5] # sigmoid
        raw_id_vector = memory_identifier.activate(self.cell_state)
        cell_state_importance = raw_id_vector[0]
        id_vector = [int(id_tag*self.ID_TAG_RANGE) for id_tag in raw_id_vector[1:]]

        # 2
        relevant_memories = self.memories.lookup(id_vector)

        # 3
        if cell_state_importance >= self.MEMORY_IMPORTANCE_THRESHOLD:
            self.memories.storeMemory(self.cell_state, id_vector)

        # 4
        total_importance = sum(importance for (memory, importance) in relevant_memories)
        scaled_memories = [tuple(element * importance/total_importance for element in memory) for (memory, importance) in relevant_memories]
        context_vector = tuple(sum(element_values) for element_values in zip(*scaled_memories)) # note: zip(*array) is a transpose operation

        # 5
        lstm_output = LSTM.activate(self, inputs+context_vector)

        return lstm_output

    

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (an LSTM). """

        # HARDCODE EVERYTHING YEEEEHAAAW

        num_top_level_inputs = config.genome_config.num_inputs # the number of inputs the LSTM appears to have to outside observers
        num_top_level_outputs = config.genome_config.num_outputs # the number of outputs the LSTM appears to have to outside observers
        internal_num_inputs = num_top_level_inputs + num_top_level_outputs # the number of inputs after secretly concatenating extra values to the outsiders' given input

        layer_details = [
            ('sigmoid', internal_num_inputs, CELL_STATE_LENGTH), # cell state managers
            ('sigmoid', internal_num_inputs, CELL_STATE_LENGTH),
            ('tanh', internal_num_inputs, CELL_STATE_LENGTH),

            ('sigmoid', internal_num_inputs, num_top_level_outputs), # outputters
            ('tanh', CELL_STATE_LENGTH, num_top_level_outputs),

            ('sigmoid', CELL_STATE_LENGTH, ID_VECTOR_LENGTH+1), # id vector labeller / memory manager # +1 for also outputting importance
        ]

        lstm = LSTM.create_given_layer_details(genome, config, layer_details, constructor=LSTM_WithMemory)
        if hasattr(genome, 'memories'):
            lstm.memories = genome.memories
        else:
            genome.memories = lstm.memories

        return lstm




# not technically part of an LSTM, but required for my implementation
class MemoryDB:
    allMemories = list()
    memoryLookup = list() # of form: [idIndex: { idValue1: {memories}, idValue2: {memories}, ... }, ...]
                        # where `memories` is of form `(memory, importance), ...`

    idVectorLen = None
    maxNumMemories = 100

    numMemoriesReturnedByLookup = None

    def __init__(self, idVectorLen = 10, numMemoriesReturnedByLookup = 5):
        self.idVectorLen = idVectorLen
        self.numMemoriesReturnedByLookup = numMemoriesReturnedByLookup

        for i in range(idVectorLen):
            self.memoryLookup.append(dict())

    @staticmethod
    def reproduce(p1, p2):
        new = MemoryDB()
        new.maxNumMemories = p1.maxNumMemories
        new.idVectorLen = p1.idVectorLen
        new.numMemoriesReturnedByLookup = p1.numMemoriesReturnedByLookup

        parents_memories_interleaved = [mem for pair in zip(p1.allMemories, p2.allMemories) for mem in pair] 
        starting_memories = parents_memories_interleaved[0:new.maxNumMemories]
        
        [new.storeMemory(*mem) for mem in starting_memories]

        return new


    def storeMemory(self, memory, idVector, importance):
        self.allMemories.append((memory, idVector, importance))

        for i, tag in zip(range(self.idVectorLen), idVector):
            if memoryLookup[i][tag] == None:
                memoryLookup[i][tag] = set()
            memoryLookup[i][tag].add((memory, importance))

        # if we have too many memories, remove the oldest as extras
        if len(self.allMemories) > self.maxNumMemories:
            (memToRemove, idVecToRemove) = self.allMemories.pop(0)
            
            for i, tag in zip(range(self.idVectorLen), idVecToRemove):
                memoryLookup[i][tag].remove(memToRemove)

    def lookup(self, idVector):
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

    
        