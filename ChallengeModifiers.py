
class ChallengeModifier:

	def pre_gen_start(self, world, new_agents):
		pass

	def on_world_iteration(self, world, agents):
		pass


class ChallengeModifierKeyFrame:
	def 





warmup_challenge = ChallengeModifier
(
	lambda self, world, new_agents:
		for agent in new_agents:
			self.get_stat('food_distance')
			self.get_stat('food_theta_range')
			self.get_stat('food_bonus_distance_range')

			# spawn food
	,
	None,
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






