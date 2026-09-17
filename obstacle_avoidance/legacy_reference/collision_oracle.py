"""
Collision Oracle for Aloha Environments.
Inspects MuJoCo contact manifold to detect ground-truth physical collisions with obstacles.
Strictly decoupled from safety filter interventions.
"""

from typing import Tuple, List, Set

class CollisionOracle:
    def __init__(self, obstacle_geom_names: Set[str] = None):
        if obstacle_geom_names is None:
            obstacle_geom_names = {"obstacle_geom", "obstacle_geom_moving", "obstacle_box", "obstacle_cylinder"}
        self.obstacle_geom_names = obstacle_geom_names
        
        # Allowed whitelist contacts
        self.whitelist_substrings = [
            ("finger", "finger"),      # Gripper fingers touching
            ("finger", "box"),         # Gripper grasping box/cube
            ("finger", "red_box"),     # Red box grasp
            ("finger", "peg"),         # Peg grasp
            ("finger", "socket"),      # Socket insertion
            ("box", "table"),          # Object resting on table
            ("red_box", "table"),
            ("peg", "socket"),         # Target insertion contact
            ("table", "table"),
        ]

    def check_collision(self, physics) -> Tuple[bool, List[dict]]:
        """
        Inspects physics.data.contact.
        Returns (has_collision, collision_details).
        """
        ncon = physics.data.ncon
        has_collision = False
        collision_details = []

        for i in range(ncon):
            con = physics.data.contact[i]
            geom1_name = physics.model.geom(con.geom1).name
            geom2_name = physics.model.geom(con.geom2).name
            dist = float(con.dist)

            # Check if any contact is with obstacle
            is_obstacle_contact = (
                geom1_name in self.obstacle_geom_names or 
                geom2_name in self.obstacle_geom_names or
                "obstacle" in geom1_name.lower() or 
                "obstacle" in geom2_name.lower()
            )

            if is_obstacle_contact and dist < 0.0:  # Penetration
                # Check whitelist
                whitelisted = False
                for w1, w2 in self.whitelist_substrings:
                    if (w1 in geom1_name.lower() and w2 in geom2_name.lower()) or \
                       (w2 in geom1_name.lower() and w1 in geom2_name.lower()):
                        whitelisted = True
                        break
                
                if not whitelisted:
                    has_collision = True
                    collision_details.append({
                        "geom1": geom1_name,
                        "geom2": geom2_name,
                        "dist": dist
                    })

        return has_collision, collision_details
