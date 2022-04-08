
def avg(generator):
	c = 0
	s = 0
	for elem in generator:
		c += 1
		s += elem
	return s/c

class ChallengeModifier:

	def __init__(trigger_on_gen_start, trigger_on_world_iteration, trigger_on_gen_end, keyframes):
		self.trigger_on_gen_start = trigger_on_gen_start
		self.trigger_on_world_iteration = trigger_on_world_iteration
		self.trigger_on_gen_end = trigger_on_gen_end
		self.keyframes = keyframes

		self.transitioning = False
		self.keyframe = self.keyframes[0]
		self.next_keyframe = self.keyframes[1]
		self.keyframe_index = 0

	def pre_gen_start(self, world, new_agents):
		self.trigger_on_gen_start(world, new_agents)

	def on_world_iteration(self, world, agents):
		self.trigger_on_world_iteration(world, agents)

	def on_gen_end(self, world, agents):
		self.trigger_on_gen_end(world, agents)

		#
		# handle keyframes
		#
		if self.transitioning:
			self.transition progress += 1

			if self.transition_progress >= self.transition_over:
				self.transitioning = False
				self.keyframe_index += 1
				self.keyframe = self.keyframes[self.keyframe_index]
				self.next_keyframe = self.keyframes[self.keyframe_index+1]
				
		if self.keyframe_index == len(self.keyframes)-1:
			return

		if avg(agent.fitness for agent in agents) > self.keyframe.transition_after_avg_fitness,
			self.transitioning = true
			self.transition progress = 1
			self.transition_max = self.keyframe.transition_over + 1
			

	def get_stat(self, stat_name):
		if not self.transitioning:
			return self.keyframe[stat_name]
		else:
			interpolation = self.transition_progress / self.transition_max
			return self.keyframe[stat_name]*(1-interpolation) + self.next_keyframe[stat_name]*interpolation


import world
import math


warmup_challenge = ChallengeModifier
(
	lambda self, world, new_agents:
		self.foods = []

		for agent in new_agents:
			dist = self.get_stat('food_distance')
			dtheta = self.get_stat('food_theta_range')
			dd = self.get_stat('food_bonus_distance_range')

			dist += math.random()*dd
			theta = (math.random()*2-0.5)*dtheta

			x = agent.x+dist*math.cos(theta)
			y = agent.y+dist*math.sin(theta)

			food_circle = QuadTree.Circle(QuadTree.Point(x, y), FOOD_RADIUS)
			food_circle.isFood = True
			self.foods.append(food_circle)
			world.quadtree.add(food_circle)
	,
	None,
	lambda self, world, agents:
		for food in self.foods:
			world.quadtree.remove(food)
	,
	[
		{
			'transition_after_avg_fitness': world.FOOD_VALUE * 0.7, # ie a majority reach eat one food
			'transition_over': 10, # take 10 gens to make the transition to the next keyframe

			'food_distance': world.AGENT_RADIUS+world.FOOD_RADIUS,
			'food_theta_range' : 0,
			'food_bonus_distance_range' : 0,

			'chance_to_activate': 1
		},
		{
			'transition_after_avg_fitness': world.FOOD_VALUE * 0.7, # ie a majority reach eat one food
			'transition_over': 20, # take 10 gens to make the transition

			'food_distance': world.AGENT_RADIUS+world.FOOD_RADIUS,
			'food_theta_range' : math.PI/8, # put the food at an angle between (-pi/8, pi/8) in front of each agent
			'food_bonus_distance_range' : 0,

			'chance_to_activate': 1
		},
		{
			'transition_after_avg_fitness': world.FOOD_VALUE * 0.7, # ie a majority reach eat one food
			'transition_over': 10, # take 10 gens to make the transition

			'food_distance': world.AGENT_RADIUS+world.FOOD_RADIUS,
			'food_theta_range' : math.PI,
			'food_bonus_distance_range' : 0,

			'chance_to_activate': 1
		},
		{
			'transition_after_avg_fitness': world.FOOD_VALUE * 0.7, # ie a majority reach eat one food
			'transition_over': 10, # take 10 gens to make the transition 

			'food_distance': world.AGENT_RADIUS+world.FOOD_RADIUS,
			'food_theta_range': math.PI,
			'food_bonus_distance_range': world.AGENT_RADIUS*5,

			'chance_to_activate': 1
		},
		{
			'chance_to_activate': 0, 

			'food_distance': world.AGENT_RADIUS+world.FOOD_RADIUS,
			'food_theta_range': math.PI,
			'food_bonus_distance_range': world.AGENT_RADIUS*5
		}
	]
)


# TODO: somehow add a "begin after this other modifier is on last frame"





ALL_MODIFIERS = [
	warmup_challenge
]