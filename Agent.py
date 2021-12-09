
import random
import math

# graph ops:
MOVE_CURR_TO_NEXT = 0
MOVE_CURR_TO_PREV = 1
MOVE_NEXT_TO_NEXT_CHILD = 2
MOVE_NEXT_TO_NEW_NODE = 3 # also creates connection to the new node

MAKE_CONNECTION_FROM_PREV = 4
MAKE_CONNECTION_TO_PREV = 5
REMOVE_CONNECTION_TO_NEXT = 6
REMOVE_CONNECTION_FROM_PREV = 7

JUMP_TO_NODE_WITH_LABEL = 8
JUMP_TO_NODE_LABEL = 9 # this node is only used for the above node
SET_CURR_LABEL = 10
NEW_LABEL = 11 # this node is only used for the above node


RESERVED_OUTPUT_COUNT = NEW_LABEL+1

# inputs:
# CURR_LABEL
# NEXT_LABEL
# PREV_LABEL

RESERVED_INPUT_COUNT = 3


def find_index_of_closest_value(array, value):
	# code from https://stackoverflow.com/a/9706105/9643841
	return min(range(len(array)), key=lambda i: abs(array[i]-value)) 

class GraphBrain:
	def __init__(self, net):
		self.net = net
		self.graph = [] # encoded as adjacency list
		self.graph_labels = []
		
		self.prev = None
		self.curr = None
		self.next = None

		# create initial knowledge base graph
		for i in range(100):
			fake_outside_world_input = [0]*AGENT_INPUT_COUNT
			self.evaluate(fake_outside_world_input)

	def evaluate(self, outside_world_input):
		graph_input = [(self.graph_labels[i] if i != None else 0) for i in [self.prev, self.curr, self.next]]
		input_vector = graph_input + outside_world_input
		output_vector = self.net.activate(input_vector)

		# do graph operations
		chosen_operation = -1
		max_confidence = -1
		for i in range(RESERVED_OUTPUT_COUNT):
			if i == JUMP_TO_NODE_LABEL or i == NEW_LABEL:
				continue
			
			if output_vector[i] > max_confidence:
				max_confidence = output_vector[i]
				chosen_operation = i

		if chosen_operation >= 0:
			self.do_graph_operation(chosen_operation, output_vector)

		return output_vector[RESERVED_OUTPUT_COUNT:]


	def __random_choice_from_children(self, node):
		if len(self.graph[node]) <= 0:
			return node

		return random.choice(self.graph[node])

	def do_graph_operation(self, chosen_operation, output_vector):
		if chosen_operation == MOVE_CURR_TO_NEXT:
			if self.next == None:
				return

			self.prev = self.curr
			self.curr = self.next
			self.next = self.__random_choice_from_children(self.curr)
		elif chosen_operation == MOVE_CURR_TO_PREV:
			if self.prev == None:
				return

			temp = self.curr
			self.curr = self.prev
			self.prev = temp
			self.next = self.__random_choice_from_children(self.curr)

		elif chosen_operation == MOVE_NEXT_TO_NEXT_CHILD:
			if self.curr == None:
				return

			children = self.graph[self.curr]
			try:
				self.next = children[(children.index(self.next)+1)%len(children)]
			except:
				self.next = self.__random_choice_from_children(self.curr)  # if next isn't pointing to a child of curr, pick a random child of curr
		
		elif chosen_operation == MOVE_NEXT_TO_NEW_NODE:
			self.next = len(self.graph)
			self.graph.append([])
			self.graph_labels.append(0)

			if self.curr == None:
				self.curr = self.next
			else:
				self.graph[self.curr].append(self.next)

		elif chosen_operation == MAKE_CONNECTION_FROM_PREV:
			if self.prev == None or self.curr == None:
				return
			self.graph[self.prev].append(self.curr)
		elif chosen_operation == MAKE_CONNECTION_TO_PREV:
			if self.prev == None or self.curr == None:
				return
			self.graph[self.curr].append(self.prev)

		elif chosen_operation == REMOVE_CONNECTION_TO_NEXT:
			if self.curr == None:
				return

			children = self.graph[self.curr]
			try:
				index = children.index(self.next)
				self.graph[self.curr].pop(index)
			except:
				pass # no connection exists
		elif chosen_operation == REMOVE_CONNECTION_FROM_PREV:
			if self.prev == None:
				return

			children = self.graph[self.prev]
			try:
				index = children.index(self.curr)
				self.graph[self.prev].pop(index)
			except:
				pass # no connection exists

		elif chosen_operation == JUMP_TO_NODE_WITH_LABEL:
			try:
				label = output_vector[JUMP_TO_NODE_LABEL]
				# find closest value in graph_labels to `label`
				found_node = find_index_of_closest_value(self.graph_labels, label)
				
				self.prev = self.curr
				self.curr = found_node
				self.next = self.__random_choice_from_children(self.curr)
			except:
				pass
		elif chosen_operation == SET_CURR_LABEL:
			if self.curr == None:
				return
			self.graph_labels[self.curr] = output_vector[NEW_LABEL]



# output
ACTION_TURN = 0
ACTION_MOVE = 1
ACTION_CRY_VOLUME = 2

AGENT_OUTPUT_COUNT = ACTION_CRY_VOLUME+1

TOTAL_OUTPUT_COUNT = AGENT_OUTPUT_COUNT + RESERVED_OUTPUT_COUNT

AGENT_STARTING_MAX_AGE = 100

# constants
AGENT_MAX_SPEED = 5
AGENT_MAX_BACKUP_SPEED = 1
AGENT_MAX_VOLUME = 10

AGENT_NUM_EYE_RAYS = 5
AGENT_FOV = math.pi/4

# AGENT_NUM_TOUCH_SENSORS = 10

AGENT_RADIUS = 0.5

# input
INPUT_TIME_LEFT = 0
INPUT_EYE_MAX = INPUT_TIME_LEFT+1 + AGENT_NUM_EYE_RAYS*2-1
INPUT_EAR_1 = INPUT_EYE_MAX+1
INPUT_EAR_2 = INPUT_EYE_MAX+2
# INPUT_TOUCH_MAX = INPUT_EYE_MAX+3 + AGENT_NUM_TOUCH_SENSORS-1 # ?

AGENT_INPUT_COUNT = INPUT_EAR_2+1

TOTAL_INPUT_COUNT = AGENT_INPUT_COUNT + RESERVED_INPUT_COUNT

class EyeRay:
	def __init__(self, rel_x, rel_y, dirr):
		self.rel_x = rel_x
		self.rel_y = rel_y
		self.dir = dirr
class Eye:
	def __init__(self, rel_x, rel_y, main_dir, num_rays, fov):
		self.rel_x = rel_x
		self.rel_y = rel_y
		self.ray_dirs = [i*fov + main_dir for i in [-0.5+j/num_rays for j in range(num_rays)]]
		self.rays = [EyeRay(self.rel_x, self.rel_y, dirr) for dirr in self.ray_dirs]
class Ear:
	def __init__(self, rel_x, rel_y):
		self.rel_x = rel_x
		self.rel_y = rel_y

class Agent:
	def __init__(self, net, parent1=None, parent2=None):
		self.age = 0
		self.max_age = AGENT_STARTING_MAX_AGE # this number is added to when eating food, removed from when hurt

		self.brain = GraphBrain(net)
		self.eyes = [Eye(0.25, 0.4, 0, AGENT_NUM_EYE_RAYS, AGENT_FOV), Eye(-0.25, 0.4, 0, AGENT_NUM_EYE_RAYS, AGENT_FOV)]
		self.ears = [Ear(0.5, 0), Ear(-0.5, 0)]

		if parent1 is None and parent2 is None:
			self.x = 0
			self.y = 0
			self.rotation = 0
		elif parent2 is None:
			self.x = parent1.x + (random.random()-0.5)
			self.y = parent1.y + (random.random()-0.5)
			self.rotation = parent1.rotation + (random.random()-0.5)
		else:
			if random.random() < 0.5:
				self.x = parent1.x + (random.random()-0.5)
				self.y = parent1.y + (random.random()-0.5)
				self.rotation = parent1.rotation + (random.random()-0.5)
			else:
				self.x = parent2.x + (random.random()-0.5)
				self.y = parent2.y + (random.random()-0.5)
				self.rotation = parent2.rotation + (random.random()-0.5)
		
	def create_input_vector(self, world):
		input_vector = []

		input_vector += [self.max_age - self.age]

		# eyes
		# world.cast_ray(self.x+self.eye1_x_rel, self.y+self.eye1_y_rel, self.rotation+self.)
		input_vector += [world.cast_ray(self.x+math.cos(self.rotation)*ray.rel_x, self.y+math.sin(self.rotation)*ray.rel_y, self.rotation+ray.dir) for eye in self.eyes for ray in eye.rays]

		input_vector += [world.hear(self.x+math.cos(self.rotation)*ear.rel_x, self.y+math.sin(self.rotation)*ear.rel_y) for ear in self.ears]

		# world maintains list of sound waves (origin_x, origin_y, radius, original_volume)
		# every iteration step, for every sound wave, radius += SPEED_OF_SOUND
		# if on the previous iteration, volume dropped to or below 0, it's deleted (eg if wave.volume_at_distance(radius-SPEED_OF_SOUND) <= 0)
		# world.hear(x, y):
			# r = dist(origin_xy, xy)
			# dist_to_wavefront = radius-r
			# for every wave such that dist_to_wavefront >= 0 && dist_to_wavefront < SPEED_OF_SOUND        # eg, the wave crossed this point in the current iteration
				# sum wave.volume_at_distance(r)       # def volume_at_distance(x): max(0, original_volume*(math.e^-x)-0.1)
			# return sum




		# input_vector += [world.get_collision] # ?

		return input_vector


	def evaluate(self, world):
		# think
		input_vector = self.create_input_vector(world)
		brain_output = self.brain.evaluate(input_vector)

		# act
		self.rotation += brain_output[ACTION_TURN]

		speed = max(brain_output[ACTION_MOVE], AGENT_MAX_BACKUP_SPEED/AGENT_MAX_SPEED) * AGENT_MAX_SPEED
		self.x += math.cos(self.rotation) * speed
		self.y += math.sin(self.rotation) * speed

		if brain_output[ACTION_CRY_VOLUME] > 0:
			world.make_sound(self.x, self.y, AGENT_MAX_VOLUME * brain_output[ACTION_CRY_VOLUME])

		# age
		self.age += 1
