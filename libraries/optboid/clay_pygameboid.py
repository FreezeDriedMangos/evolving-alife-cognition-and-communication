#!/usr/bin/env python

#
# This file written by Clay
#

"""
A simple pygame engine to do some 2D rendering.
Used to display boids and obstacles at given positions
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import pygame
from math import *
import optboid_extension

class DummyObject:
	pass

def rotate_vector(v, a):
	p = DummyObject()
	p.x = v[0]
	p.y = v[1]

	return (p.x * cos(a) - p.y * sin(a), p.x * sin(a) + p.y * cos(a))

class World(object):
	"""
	world class with draw functions for entities
	"""
	
	screen = None
	screen_width = 700
	screen_height = 700
	scale = 10

	boid_template = [(0, 1), (0.5, -1), (-0.5, -1)]

	def __init__(self, swarm, offx, offy):
		self.swarm = swarm
		self.ents = swarm.boids  # this will point to a list of boids
		self.ent_size = 15.0
		self.num_ents = len(swarm.boids)
		# self.fps = clock.ClockDisplay()
		self.o_x = offx
		self.o_y = offy

		pygame.init()
		self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])

	def draw_entity(self, e):
		""" Draws a boid """
		scale = self.ent_size
		scaled_boid_points = [(p[0] * scale, p[1] * scale) for p in self.boid_template]
		rotated_boid_points = [rotate_vector(p, e.rotation-pi/2) for p in scaled_boid_points]
		boid_points = [(p[0] + e.position.x + self.o_x, p[1] + e.position.y + self.o_y) for p in rotated_boid_points]

		color = (0, 0, 0) if not isinstance(e, optboid_extension.BoidObstacle) else (100, 0, 0)
		pygame.draw.polygon(self.screen, color, boid_points)


	# def draw_grid(self):
	#     cw = self.swarm.cell_width
	#     w = cw * self.swarm.divisions
	#     for i in range(self.swarm.divisions):
	#         xy = i*cw
	#         glLoadIdentity()
	#         glBegin(GL_LINES)
	#         glColor4f(0.5, 0.5, 0.5, 0)
	#         glVertex2f(0, xy)
	#         glVertex2f(w, xy)
	#         glEnd()

	#         glBegin(GL_LINES)
	#         glColor4f(0.5, 0.5, 0.5, 0)
	#         glVertex2f(xy, 0)
	#         glVertex2f(xy, w)
	#         glEnd()

	def draw(self):
		self.screen.fill((255, 255, 255))
		
		# self.fps.draw()

		#self.draw_grid()
		for ent in self.ents:
			self.draw_entity(ent)

sim = optboid_extension.FlockAndObstacleSimulation(150, 750, 10)
world = World(sim.swarm, -25, -25)

# window = pyglet.window.Window(700, 700, vsync=False)


# @window.event
# def on_draw():
#     window.clear()
#     world.draw()


# def update(dt):
# 	sim.update(dt)
# 	world.draw()


# def idle(dt):
# 	pass

# clock.schedule(update)
# clock.schedule(idle)

# if __name__ == '__main__':
#     pyglet.app.run()

if __name__ == '__main__':
	clock = pygame.time.Clock()
	
	crashed = False

	while not crashed:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				crashed = True

		pygame.display.update()
		dt = clock.tick(30)
		
		sim.update(dt/400)
		world.draw()
