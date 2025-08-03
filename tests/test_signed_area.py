import unittest
from utils import polygon_signed_area


class TestPolygonSignedArea(unittest.TestCase):
	"""
	Tests for polygon_signed_area.
	The signed area is positive for counter-clockwise (CCW) orientation,
	negative for clockwise (CW), and zero for degenerate (colinear) polygons.
	The absolute value is the area of the polygon.
	"""

	def test_triangle_ccw(self):
		# CCW triangle: (0,0) -> (1,0) -> (0,1)
		pts = [(0, 0), (1, 0), (0, 1)]
		area = polygon_signed_area(pts)
		self.assertAlmostEqual(area, 0.5)
		self.assertGreater(area, 0)

	def test_triangle_cw(self):
		# CW triangle: (0,0) -> (0,1) -> (1,0)
		pts = [(0, 0), (0, 1), (1, 0)]
		area = polygon_signed_area(pts)
		self.assertAlmostEqual(area, -0.5)
		self.assertLess(area, 0)

	def test_square_ccw(self):
		pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
		area = polygon_signed_area(pts)
		self.assertAlmostEqual(area, 1.0)
		self.assertGreater(area, 0)

	def test_square_cw(self):
		pts = [(0, 0), (0, 1), (1, 1), (1, 0)]
		area = polygon_signed_area(pts)
		self.assertAlmostEqual(area, -1.0)
		self.assertLess(area, 0)

	def test_colinear(self):
		pts = [(0, 0), (1, 0), (2, 0)]
		area = polygon_signed_area(pts)
		self.assertAlmostEqual(area, 0.0)

	def test_bowtie_self_intersecting(self):
		# Bowtie: (0,0)-(1,1)-(0,1)-(1,0)
		pts = [(0, 0), (1, 1), (0, 1), (1, 0)]
		area = polygon_signed_area(pts)
		self.assertAlmostEqual(area, 0.0)

if __name__ == "__main__":
	unittest.main()