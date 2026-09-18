import numpy as np

class DepthObstacleConstraint:
    """Conservative point-cloud safety gate; returns stop/allow only."""
    def __init__(self, min_distance_m=0.08): self.min_distance_m=float(min_distance_m)
    def evaluate(self, tool_position, obstacle_points):
        p=np.asarray(tool_position,dtype=float).reshape(3); q=np.asarray(obstacle_points,dtype=float)
        if q.size==0: return {'safe':True,'stop':False,'min_distance_m':None,'reason':'no_obstacles'}
        q=q.reshape(-1,3); d=np.linalg.norm(q-p,axis=1); md=float(d.min()); stop=md<self.min_distance_m
        return {'safe':not stop,'stop':stop,'min_distance_m':md,'reason':'obstacle_too_close' if stop else 'clear'}
    def evaluate_trajectory(self, trajectory, obstacle_points):
        """Check every waypoint; conservative gate for planned tool motion."""
        checks=[self.evaluate(p,obstacle_points) for p in np.asarray(trajectory,dtype=float).reshape(-1,3)]
        bad=[c for c in checks if c['stop']]
        return {'safe':not bad,'stop':bool(bad),'min_distance_m':min((c['min_distance_m'] for c in checks if c['min_distance_m'] is not None),default=None),'reason':'trajectory_obstacle' if bad else 'trajectory_clear'}

def inside_workspace(point, bounds):
    p=np.asarray(point,dtype=float).reshape(3)
    return all(float(lo)<=float(x)<=float(hi) for x,(lo,hi) in zip(p,bounds))
