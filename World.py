
import math
import random

import libraries.perlin_numpy as perlin_numpy

class SoundWave:
	def __init__(self, origin_x, origin_y, radius, original_volume):
		self.origin_x = origin_x
		self.origin_y = origin_y
		self.radius = radius
		self.original_volume = original_volume

		# world maintains list of sound waves (origin_x, origin_y, radius, original_volume)
		# every iteration step, for every sound wave, radius += SPEED_OF_SOUND
		# if on the previous iteration, volume dropped to or below 0, it's deleted (eg if wave.volume_at_distance(radius-SPEED_OF_SOUND) <= 0)
		# world.hear(x, y):
			# r = dist(origin_xy, xy)
			# dist_to_wavefront = radius-r
			# for every wave such that dist_to_wavefront >= 0 && dist_to_wavefront < SPEED_OF_SOUND        # eg, the wave crossed this point in the current iteration
				# sum wave.volume_at_distance(r)       # def volume_at_distance(x): max(0, original_volume*(math.e^-x)-0.1)
			# return sum

	def volume_at_distance(self, x): 
		if x < 0:
			return 0
		return max(0, self.original_volume*(math.e**(-x))-0.1)


SPEED_OF_SOUND = 4#2 #20

import numpy as np
import pprint
class World:
	def __init__(self):
		self.sound_waves = []
		self.agents = []
		self.world_min_x = 0 
		self.world_min_y = 0 
		self.world_max_x = 50 
		self.world_max_y = 50 
		self.wall_size = 0.5

		self.num_gens_of_unique_walls = 20
		self.noise = perlin_numpy.generate_perlin_noise_3d((math.ceil(self.world_max_x/self.wall_size), math.ceil(self.world_max_y/self.wall_size), self.num_gens_of_unique_walls), (2, 2, 1), tileable=(False, False, True))

		
		def normalized(a, axis=-1, order=2):
			l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
			l2[l2==0] = 1
			return a / np.expand_dims(l2, axis)
		self.noise = normalized(self.noise, axis=0, order=3)
		self.noise = normalized(self.noise, axis=1, order=3)
		self.noise = normalized(self.noise, axis=2, order=3)


		self.update_walls(0)

	def update_walls(self, time):
		time = (time//1) % self.num_gens_of_unique_walls
		self.walls = [[self.noise[x, y, time]+0.5 > 0.8 for x in range(0, math.ceil(self.world_max_x/self.wall_size))] for y in range(0, math.ceil(self.world_max_y/self.wall_size))]
		# pprint.pprint(self.walls)

	def init_agents(self, agents):
		# remove current agents from the simulation
		# for agent in self.agents: self.clear_agent(agent)

		self.agents = agents

	def iteration(self):
		# if a wave dropped below 0 volume at its wavefront last iteration, drop it
		# this is done here so that a wave that goes from 10 volume to 0 on a given iteration has a chance to be heared by listeners who would percieve it as greater than 0 volume
		# ie when a wave goes from volume 10 to 0, it technically was all volumes between 10 and 0 that iteration. The next iteration it will be 0 for the entire iteration and can be removed
		self.sound_waves = [wave for wave in self.sound_waves if wave.volume_at_distance(wave.radius) > 0]
		for wave in self.sound_waves:
			wave.radius += SPEED_OF_SOUND

		for agent in self.agents:
			agent.p_x = agent.x
			agent.p_y = agent.y

			agent.evaluate(self)
			agent.x = max(self.world_min_x, min(agent.x, self.world_max_x))
			agent.y = max(self.world_min_y, min(agent.y, self.world_max_y))

	def make_sound(self, x, y, volume):
		self.sound_waves.append(SoundWave(x, y, 0, volume))

	def hear(self, x, y):
		s = 0
		for wave in self.sound_waves:
			r = math.dist([wave.origin_x, wave.origin_y], [x, y])
			dist_from_wavefront = wave.radius-r
			s += wave.volume_at_distance(dist_from_wavefront)

			# r = dist(origin_xy, xy)
			# dist_to_wavefront = radius-r
			# for every wave such that dist_to_wavefront >= 0 && dist_to_wavefront < SPEED_OF_SOUND        # eg, the wave crossed this point in the current iteration
				# sum wave.volume_at_distance(r)       # def volume_at_distance(x): max(0, original_volume*(math.e^-x)-0.1)
			# return sum
		return s

	def cast_ray(self, x, y, dir):
		return 0

	





# ################################################
# 
#  My modified version of https://www.shadertoy.com/view/MtcGRl
#    reference for changing obstacles over time (note: it's wayyyy too fast though)
# 
# ################################################

def GetGradient(intPos, t): 
    
    # // Uncomment for calculated rand
    # //float rand = fract(math.sin(dot(intPos, vec2(12.9898, 78.233))) * 43758.5453);;
    
    # // Texture-based rand (a bit faster on my GPU)
    # float rand = texture(iChannel0, intPos / 64.0).r;
    rand = random.random()

    # // Rotate gradient: random starting rotation, random rotation rate
    angle = 6.283185 * rand + 4.0 * t * rand
    return (math.cos(angle), math.sin(angle))


def mix(x, y, a):
	return x*(1-a) + y*a

def dot(v, u):
	return sum(v_i*u_i for (v_i, u_i) in zip(v, u))

def Pseudo3dNoise(pos, time, scale=1):
	pos = tuple(pos_i*scale for pos_i in pos)

	time += 1
	i = tuple(math.floor(pos_i) for pos_i in pos)
	f = tuple(pos_i-i_i for (pos_i, i_i) in zip(pos, i))
	blend = tuple(f_i * f_i * (3-2*f_i) for f_i in f) #f * f * (3.0 - 2.0 * f);

	noiseVal = \
		mix(
			mix(
				dot(GetGradient( (i_i + v_i for (i_i, v_i) in zip(i, (0, 0))) , time), (f_i - v_i for (f_i, v_i) in zip(f, (0, 0)))),
				dot(GetGradient( (i_i + v_i for (i_i, v_i) in zip(i, (1, 0))) , time), (f_i - v_i for (f_i, v_i) in zip(f, (1, 0))) ),
				blend[0]),
			mix(
				dot(GetGradient( (i_i + v_i for (i_i, v_i) in zip(i, (0, 1))) , time), (f_i - v_i for (f_i, v_i) in zip(f, (0, 1))) ),
				dot(GetGradient( (i_i + v_i for (i_i, v_i) in zip(i, (1, 1))) , time), (f_i - v_i for (f_i, v_i) in zip(f, (1, 1))) ),
				blend[0]),
		blend[1]
	)
	normalNoiseVal = noiseVal / 0.7
	
	steppedNoiseVal = math.floor(2 * normalNoiseVal)/2 # // normalize to about [-1..1]
	return math.floor(steppedNoiseVal+0.9)


# vec2 GetGradient(vec2 intPos, float t) {
    
#     // Uncomment for calculated rand
#     //float rand = fract(math.sin(dot(intPos, vec2(12.9898, 78.233))) * 43758.5453);;
    
#     // Texture-based rand (a bit faster on my GPU)
#     float rand = texture(iChannel0, intPos / 64.0).r;
    
#     // Rotate gradient: random starting rotation, random rotation rate
#     float angle = 6.283185 * rand + 4.0 * t * rand;
#     return vec2(math.cos(angle), math.sin(angle));
# }


# float Pseudo3dNoise(vec3 pos) {
#     vec2 i = math.floor(pos.xy);
#     vec2 f = pos.xy - i;
#     vec2 blend = f * f * (3.0 - 2.0 * f);
#     float noiseVal = 
#         mix(
#             mix(
#                 dot(GetGradient(i + vec2(0, 0), pos.z), f - vec2(0, 0)),
#                 dot(GetGradient(i + vec2(1, 0), pos.z), f - vec2(1, 0)),
#                 blend.x),
#             mix(
#                 dot(GetGradient(i + vec2(0, 1), pos.z), f - vec2(0, 1)),
#                 dot(GetGradient(i + vec2(1, 1), pos.z), f - vec2(1, 1)),
#                 blend.x),
#         blend.y
#     );
#     float normalNoiseVal = noiseVal / 0.7;
    
#     float steppedNoiseVal = math.floor(2. * normalNoiseVal)/2.; // normalize to about [-1..1]
#     return math.floor(steppedNoiseVal+0.9);
# }


# void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    
# 	vec2 uv = fragCoord.xy / iResolution.y;
    
#     // Mouse up: show one noise channel
#     if (iMouse.z <= 0.0) {
#         float noiseVal = 0.5 + 0.5 * Pseudo3dNoise(vec3(uv * 10.0, iTime));
#         fragColor.rgb = vec3(noiseVal);
#     }
    
# #     // Mouse down: layered noise
# #     else {
# # 		const int ITERATIONS = 10;
# #         float noiseVal = 0.0;
# #         float sum = 0.0;
# #         float multiplier = .0000000001;
# #         for (int i = 0; i < ITERATIONS; i++) {
# #             vec3 noisePos = vec3(uv, 0.2 * iTime / multiplier);
# #             noiseVal += multiplier * abs(Pseudo3dNoise(noisePos));
# #             sum += multiplier;
# #             multiplier *= 0.6;
# #             uv = 2.0 * uv + 4.3;
# #         }
# #         noiseVal /= sum;
        
# #         // Map to a color palette
# #         fragColor.rgb = 0.5 + 0.5 * math.cos(6.283185 * (3.0 * noiseVal + vec3(0.15, 0.0, 0.0)));
# #     }
# }