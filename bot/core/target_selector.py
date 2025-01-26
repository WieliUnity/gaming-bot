# bot/core/target_selector.py
import numpy as np
import time

class TargetSelector:
    def __init__(self, cluster_threshold=100, lock_duration=5):
        # Clustering parameters
        self.cluster_threshold = cluster_threshold  # Pixel distance for grouping trees
        # Target locking parameters
        self.current_target = None
        self.target_lock_duration = lock_duration
        self.last_lock_time = 0

    def cluster_trees(self, boxes):
        """Groups nearby trees into clusters using simple distance-based grouping."""
        clusters = []
        used_indices = set()

        for i, (x1, y1, w1, h1) in enumerate(boxes):
            if i in used_indices:
                continue
            cluster = [i]
            used_indices.add(i)
            center1 = np.array([x1 + w1/2, y1 + h1/2])

            for j, (x2, y2, w2, h2) in enumerate(boxes):
                if j in used_indices:
                    continue
                center2 = np.array([x2 + w2/2, y2 + h2/2])
                distance = np.linalg.norm(center1 - center2)

                if distance < self.cluster_threshold:
                    cluster.append(j)
                    used_indices.add(j)

            clusters.append(cluster)
        return clusters

    def select_target(self, boxes):
        # Check target lock status
        if self.current_target and (time.time() - self.last_lock_time < self.target_lock_duration):
            return self.current_target
        
        # Find clusters if no active target
        clusters = self.cluster_trees(boxes)
        if not clusters:
            return None

        # Find largest cluster
        largest_cluster = max(clusters, key=lambda x: len(x))
        
        # Find largest tree in cluster
        largest_area = 0
        best_box = None
        for idx in largest_cluster:
            x, y, w, h = boxes[idx]
            area = w * h
            if area > largest_area:
                largest_area = area
                best_box = boxes[idx]

        # Update target lock
        self.current_target = best_box
        self.last_lock_time = time.time()
        return best_box