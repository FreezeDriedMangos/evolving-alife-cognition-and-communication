
import pygame
import sys
import math


# draw each creature
# draw each soundwave, kinda like this:
# -=%#
#   -=%#
#    -=%#
#     -=%#
#     -=%#
#    -=%#
#   -=%#
# -=%#

has_made_plot = False



screen = None
screen_width = 500
screen_height = 500
scale = 10
def init():
	global screen

	pygame.init()
	# pygame.display.list_modes()

	# Set up the drawing window
	screen = pygame.display.set_mode([500, 500])

def draw(world):
	screen.fill((255, 255, 255))

	for agent in world.agents:
		pygame.draw.circle(screen, (100, 100, 100), (agent.x*scale, agent.y*scale), 1*scale)

	for sound_wave in world.sound_waves:
		volume = int(sound_wave.volume_at_distance(sound_wave.radius))
		color = (0,0,0)
		if volume == 0:
			volume = 1
			color = (50, 50, 50)
		pygame.draw.circle(screen, color, (sound_wave.origin_x*scale, sound_wave.origin_y*scale), sound_wave.radius*scale, width=volume)

	for y in range(0, math.ceil(world.world_max_y/world.wall_size)):
		for x in range(0, math.ceil(world.world_max_x/world.wall_size)):
			if world.walls[x][y]:
				pygame.draw.rect(screen, (0, 0, 0), (world.wall_size*x*scale, world.wall_size*y*scale, world.wall_size*scale, world.wall_size*scale))
			# color = tuple(int(255*(world.walls[x][y])%1) for i in range(3))
			# pygame.draw.rect(screen, color, (world.wall_size*x*scale, world.wall_size*y*scale, world.wall_size*scale, world.wall_size*scale))

	# Flip the display
	pygame.display.flip()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit(0)


	# global has_made_plot
	# if not has_made_plot:
	# 	has_made_plot = True

	# 	import matplotlib.pyplot as plt

	# 	import pymunk
	# 	import pymunk.matplotlib_util
	# 	from pymunk.vec2d import Vec2d

	# 	space = world.space

	# 	fig = plt.figure(figsize=(14, 10))
	# 	ax = plt.axes(xlim=(0, 150), ylim=(0, 150))
	# 	ax.set_aspect("equal")
	# 	o = pymunk.matplotlib_util.DrawOptions(ax)
	# 	space.debug_draw(o)

	# 	fig.savefig("matplotlib_util_demo.png", bbox_inches="tight")