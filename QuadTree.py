import math

class Vector():
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __add__(self, o):
		return Vector(self.x + o.y, self.y + o.y)

	def __sub__(self, o):
		return Vector(self.x - o.y, self.y - o.y)

	def __mul__(self, s):
		return Vector(self.x*s, self.y*s)
	
	def __truediv__(self, s):
		return Vector(self.x/s, self.y/s)	

	def magnitude(self):
		return math.sqrt(self.x*self.x + self.y*self.y)

	def normalized(self):
		return self/self.magnitude()

class Point(Vector):
	pass



class BoundingBox():
	left = 0
	top = 0
	right = 0
	bottom = 0

	def __init__(self, left=0, top=0, right=0, bottom=0):
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom

		self._inner_rad = None
		self._outer_rad = None 
		self._center = None

	def width(self):
		return self.right-self.left
	
	def height(self):
		return self.bottom-self.top
		
	def inner_rad(self):
		if self._inner_rad == None:
			self._inner_rad = min(self.width()/2, self.height()/2)
		return self._inner_rad

	def outer_rad(self):
		if self._outer_rad == None:
			center_to_corner = Vector(self.width()/2, self.height()/2)
			self._outer_rad = math.sqrt(center_to_corner.x*center_to_corner.x + center_to_corner.y*center_to_corner.y)
		return self._outer_rad

	def center(self):
		if self._center == None:
			corner_to_center = Vector(self.width()/2, self.height()/2)
			self._center = Point(self.left, self.top) + corner_to_center
		return self._center


class Ray():
	def __init__(self, origin, direction):
		self.dir_inv = tuple(1/dir for dir in direction)
		self.direction = direction
		self.origin = origin

class Circle():
	def __init__(self, origin, radius):
		self.origin = origin
		self.radius = radius


class CollisionDetector():

	#
	# main function
	#
 
	def collides(self, obj1, obj2):
		collisionFunctions = [
			(BoundingBox, BoundingBox, self.boxbox),
			(BoundingBox, Ray, self.raybox),
			(BoundingBox, Circle, self.boxcircle),

			(Ray, Ray, None),
			(Ray, Circle, self.raycircle),
			
			(Circle, Circle, self.circlecircle),
		]

		for (type1, type2, func) in collisionFunctions:
			if isinstance(obj1, type1) and isinstance(obj1, type2):
				return func(obj1, obj2)
			if isinstance(obj2, type1) and isinstance(obj1, type2):
				return func(obj2, obj1)

		raise

	#
	# util
	#

	def distanceSquared(self, p1, p2):
		d = p1-p2
		return d.x*d.x + d.y*d.y

	#
	# Specific collision detection functions
	#
 
	# https://tavianator.com/2011/ray_box.html
	def raybox(self, r, b):
		tx1 = (b.left - r.origin.x)*r.dir_inv.x
		tx2 = (b.right - r.origin.x)*r.dir_inv.x

		tmin = min(tx1, tx2)
		tmax = max(tx1, tx2)

		ty1 = (b.top - r.origin.y)*r.dir_inv.y
		ty2 = (b.bottom - r.origin.y)*r.dir_inv.y

		tmin = max(tmin, min(ty1, ty2))
		tmax = min(tmax, max(ty1, ty2))

		# return tmax >= tmin
		return tmax >= 0

	def circlecircle(self, c1, c2):
		return self.distanceSquared(c1.origin, c2.origin) <= (c1.radius+c2.radius)*(c1.radius+c2.radius)

	def raycircle(self, r, c):
		# https://stackoverflow.com/a/1084899/9643841
		# https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
		D = r.dir
		L = c.origin - r.origin
		t_ca = L.x*D.x + L.y+D.y

		if (t_ca < 0):
			return False

		d_squared = (L.x*L.x+L.y*L.y) - (t_ca.x*t_ca.x+t_ca.y*t_ca.y)
		c_radius_squared = c.radius*c.radius
		if (d_squared < c_radius_squared):
			return False

		t_hc = math.sqrt(c_radius_squared - d_squared)
		t_0 = t_ca-t_hc
		t_1 = t_ca+t_hc

		return (r.origin + t_0*r.dir, r.origin + t_1*r.dir)


	def boxcircle(self, b, c):
		# https://stackoverflow.com/a/402010/9643841
		# circleDistance = Vector(
		# 	abs(c.origin.x - b.origin.x), 
		# 	abs(c.origin.y - b.origin.y)
		# )

		# if circleDistance.x > (b.width()/2 + c.radius) \
		# or circleDistance.y > (b.height()/2 + c.radius):
		# 	return False

		# if circleDistance.x <= (b.width()/2) \
		# or circleDistance.y <= (b.height()/2):
		# 	return True

		# cornerDistance_sq = (circleDistance.x - b.width()/2)^2 + \
		# 					(circleDistance.y - b.height()/2)^2

		# return cornerDistance_sq <= (c.radius^2)

		# https://gamedev.stackexchange.com/a/96544
		distance_squared = self.distanceSquared(c.origin, b.center())
		if distance_squared > (b.outer_rad() + c.radius)**2:
			return False
		if distance_squared < (b.inner_rad() + c.radius)**2:
			return True
		
		dir_from_box_to_circle = (c.origin - b.center()).normalized()
		closest_point_on_circle = c.origin + dir_from_box_to_circle*c.radius

		return b.left <= closest_point_on_circle.x and closest_point_on_circle.x <= b.right \
		   and b.top  <= closest_point_on_circle.y and closest_point_on_circle.y <= b.bottom


	def boxbox_helper(self, b1, b2):
		return b1.left <= b2.left and b2.left <= b1.right \
		   and b1.top  <= b2.top  and b2.top  <= b1.bottom

	def boxbox(self, b1, b2):
		return self.boxbox_helper(b1, b2) or self.boxbox_helper(b2, b1)


class NodeManager():
	def __init__(self, maxDepth = 5):
		self.maxDepth = maxDepth
		self.collisionDetector = CollisionDetector()
		self.availableNodes = []

	def _createNode(self):
		qn = QuadNode(self, self.collisionDetector)
		return qn

	def getOrCreateNode(self):
		if len(self.availableNodes) == 0:
			return self._createNode()

		return self.availableNodes.pop()

	def returnNode(self, node):
		self.availableNodes.append(node)



class QuadNode():
	def __init__(self, nodeManager, collisionDetector):
		self.nodeManager = nodeManager
		self.collisionDetector = collisionDetector

		self.occupants = []
		self.occupantsSet = set()
		self.children = []
		self.depth = 0

		self.boundingBox = None

	def add(self, occupant):
		self.occupants.append(occupant)
		self.occupantsSet.add(occupant)

		if self.depth < self.nodeManager.maxDepth and len(self.children) == 0:
			offsets = [(0, 0), (1, 0), (1,1), (0,1)]
			width = self.boundingBox.width()/2.0
			height = self.boundingBox.height()/2.0
			
			for (dx, dy) in offsets:
				child = self.nodeManager.getOrCreateNode()
				child.boundingBox = BoundingBox(
					self.boundingBox.left + dx*width,
					self.boundingBox.top  + dy*height,
					self.boundingBox.left + (dx+1)*width,
					self.boundingBox.top  + (dy+1)*height,
				)
				child.depth = self.depth+1
				self.children.append(child)

		for child in self.children:
			if self.collisionDetector.collides(occupant, child.boundingBox):
				child.add(occupant)
		
	def isEmpty(self):
		return len(self.occupants) == 0

	def contains(self, occupant):
		return occupant in self.occupantsSet

	def remove(self, occupant):
		if not self.contains(occupant):
			return

		self.occupants.remove(occupant)
		self.occupantsSet.remove(occupant)

		for child in self.children:
			child.remove(occupant)
		
		if all(child.isEmpty() for child in self.children):
			for child in self.children:
				self.nodeManager.returnNode(child)
			self.children.clear()

	# TODO: optimizations can happen here
	def update(self, occupant):
		self.remove(occupant)
		self.add(occupant)


	#
	# Special functions
	#

	def _getCollisions(self, checked_potential_collisions, found_collisions):
		if len(self.children) > 0:
			for child in self.children:
				child._getCollisions(checked_potential_collisions, found_collisions)
			return found_collisions
			
		for i in range(len(self.occupants)):
			for j in range(i+1, len(self.occupants)):
				collision = (self.occupants[i], self.occupants[j])
				collision_inverse = (self.occupants[j], self.occupants[i])

				if collision in checked_potential_collisions:
					continue

				checked_potential_collisions.add(collision)
				checked_potential_collisions.add(collision_inverse)

				if self.collisionDetector.collides(*collision):
					found_collisions.add(collision)
		
		return found_collisions

	def getCollisions(self):
		return self._getCollisions(set(), set())

	# returns (absolute location of collision, object collided)
	def raycast(self, ray):
		if len(self.occupants) == 0:
			return None

		if len(self.children) == 0:
			collisions = [(self.collisionDetector(ray, self.obj), obj) for obj in self.occupants]
			closestCollision = min(collisions, key=lambda collision: self.collisionDetector.distanceSquared(collision[0], ray.origin))
			return closestCollision
		
		childrenCenters = [
			(
				(child.boundingBox.right-child.boundingBox.left)/2 + child.boundingBox.left, 
				(child.boundingBox.bottom-child.boundingBox.top)/2 + child.boundingBox.top
			) 
			for child in self.children
		]

		childrenDistancesFromRayOrigin = [self.collisionDetector.distanceSquared(center, ray.origin) for center in childrenCenters]

		orderedChildrenAndDistances = zip(self.children, childrenDistancesFromRayOrigin).sort(key=lambda e: e[1])
		orderedChildren = [child for (child, distance) in orderedChildrenAndDistances]

		for child in orderedChildren:
			if not self.collisionDetector.collision(child.boundingBox, ray):
				continue

			collision = child.raycast(ray)
			if collision == None:
				continue
			
			return collision

		return None
		

class QuadTree(QuadNode):
	def __init__(self, worldWidth, worldHeight):
		QuadNode.__init__(self, NodeManager(), CollisionDetector())
		self.boundingBox = BoundingBox()
		self.boundingBox.left = 0
		self.boundingBox.top = 0
		self.boundingBox.right = worldWidth
		self.boundingBox.bottom = worldHeight
