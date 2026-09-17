import unittest, numpy as np
from obstacle_constraint import DepthObstacleConstraint, inside_workspace

class ObstacleTests(unittest.TestCase):
    def test_clear_and_stop(self):
        g=DepthObstacleConstraint(.1)
        self.assertFalse(g.evaluate([0,0,0],[[0,0,.2]])['stop'])
        self.assertTrue(g.evaluate([0,0,0],[[0,0,.05]])['stop'])
    def test_empty_depth(self): self.assertTrue(DepthObstacleConstraint().evaluate([0,0,0],[])['safe'])
    def test_trajectory_and_workspace(self):
        g=DepthObstacleConstraint(.1); r=g.evaluate_trajectory([[0,0,.2],[0,0,.05]],[] if False else [[0,0,.05]])
        self.assertTrue(r['stop']); self.assertTrue(inside_workspace([.1,0,.2],[(0,1),(-1,1),(0,1)])); self.assertFalse(inside_workspace([2,0,.2],[(0,1),(-1,1),(0,1)]))

if __name__=='__main__': unittest.main()
