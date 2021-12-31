from __future__ import division, print_function, absolute_import, unicode_literals
import libraries.optboid.simulation as optboid


import math
import operator
import types
from math import atan2, sin, cos, floor, ceil
import random
import collections

class BoidObstacle(optboid.Boid):
	def update(self, t):
		pass

class SmartBoid(optboid.Boid):
	
	_avoid_f = optboid.Vector2(0, 0)
	_nearest_obstacle_distance = float('inf')

	def interact(self, actors):
		"""
		Unit-unit interaction method, combining a separation force, and velocity
		alignment force, and a cohesion force.

		Many examples separate these into different functions for clarity
		but combining them means we need fewer loops over the neibor list
		"""
		self.actors = actors # hacky, I'm not a fan

		self._sep_f.clear()
		self._align_f.clear()
		self._cohes_sum.clear()

		count = 0
		self.neighbors = len(actors)

		for other in actors:
			# vector pointing from neighbors to self
			diff = self.position - other.position
			d = abs(diff)

			# Only perform on "neighbor" actors, i.e. ones closer than arbitrary
			# dist or if the distance is not 0 (you are yourself)
			if 0 < d < self.influence_range:

				diff.normalize()
				if d < self.minsep:
#                    diff *= self.max_force
#                else:
					diff /= d  # Weight by distance

				if not isinstance(other, BoidObstacle):
					count += 1
					self._sep_f += diff

					self._cohes_sum += other.position  # Add position

					# Align - add the velocity of the neighbouring actors, then average
					self._align_f += other.velocity


		if count > 0:
			# calc the average of the separation vector
			# self._sep_f /=count don't div by count if normalizing anyway!
			self._sep_f.normalize()
			self._sep_f *= self.max_speed
			self._sep_f -= self.velocity
			optboid.limit(self._sep_f, self.max_force)

			# calc the average direction (normed avg velocity)
			# self._align_f /= count
			self._align_f.normalize()
			self._align_f *= self.max_speed
			self._align_f -= self.velocity
			optboid.limit(self._align_f, self.max_force)

			# calc the average position and calc steering vector towards it
			self._cohes_sum /= count
			cohesion_f = self.steer(self._cohes_sum, True)

			self._sep_f *= self.sep_strength
			self._align_f *= self.align_strength
			cohesion_f *= self.cohesion_strength


			# finally add the velocities
			sum = self._sep_f + cohesion_f + self._align_f

			self.acceleration = sum

	def avoid_obstacles(self, obstacles):
		ahead = self.position + self.velocity

		self._avoid_f.clear()
		self._nearest_obstacle_distance = float('inf')

		average_obstacle_position = optboid.Vector2(0, 0)

		count = 0
		for other in obstacles:
			# vector pointing from neighbors to self
			diff = self.position - other.position
			d = abs(diff)

			# Only perform on "neighbor" actors, i.e. ones closer than arbitrary
			# dist or if the distance is not 0 (you are yourself)
			if 0 < d < self.influence_range:

				diff.normalize()
				if d < self.minsep:
#                    diff *= self.max_force
#                else:
					diff /= d  # Weight by distance

				if isinstance(other, BoidObstacle):
					count += 1

					distance = (self.position - other.position).magnitude()
					
					if distance < self._nearest_obstacle_distance:
						self._nearest_obstacle_distance = distance

						ahead = self.position + self.velocity
						self._avoid_f = ahead - other.position

					average_obstacle_position += other.position
		
		if count > 0:
			# calc the avoidance force
			average_obstacle_position /= count
			self._avoid_f = self.position + self.velocity - average_obstacle_position
			# self._avoid_f.normalize()
			# self._avoid_f *= self.max_speed
			self._avoid_f -= self.velocity
			# optboid.limit(self._avoid_f, self.max_force)

			self.acceleration = self._avoid_f

	
	def update(self, t):
		"""
		Method to update position by computing displacement from velocity and acceleration
		"""
		self.velocity += self.acceleration * t
		optboid.limit(self.velocity, self.max_speed)
		
		self.avoid_obstacles([boid for boid in self.actors if isinstance(boid, BoidObstacle)])
		self.velocity += self.acceleration * t
		optboid.limit(self.velocity, self.max_speed)

		self.position += self.velocity * t

		
class FlockAndObstacleSimulation(optboid.FlockSimulation):
	def __init__(self, starting_units=100, field_size=800, starting_obstacles=50):
		"""
		"""
		self.swarm = optboid.BoidSwarm(field_size+2*40, optboid.Boid.influence_range+5)  # /2
		self.field_size = field_size
		self.pad = 40  # use to keep boids inside the play field

		for i in range(starting_units):
			b = SmartBoid(random.uniform(100, 400),
						random.uniform(100, 400))
			self.swarm.boids.append(b)
		for i in range(starting_obstacles):
			b = BoidObstacle(random.uniform(100, 400),
							random.uniform(100, 400))
			self.swarm.boids.append(b)
		self.swarm.rebuild()
		self._cumltime = 0  # calculation var