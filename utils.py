import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

def get_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

def average_feature(track):
    vectors = [entry[3] for entry in track if len(entry) > 3 and np.linalg.norm(entry[3]) > 0.1]
    if not vectors:
        return np.zeros(512)
    vectors = sorted(vectors, key=lambda v: np.linalg.norm(v), reverse=True)[:5]
    mean_vector = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean_vector)
    return mean_vector / norm if norm > 0 else np.zeros(512)

def classify_team_color(track):
    # Simple heuristic: average R channel to guess team color
    avg_colors = []
    for entry in track:
        if len(entry) < 4:
            continue
        feat = entry[3]
        if feat is not None and np.linalg.norm(feat) > 0:
            avg_colors.append(np.mean(feat))  # crude approximation
    return 'team1' if np.mean(avg_colors) > 0 else 'team2'

def match_by_appearance(broadcast_tracks, tacticam_tracks):
    broadcast_features = {
        tid: average_feature(track)
        for tid, track in broadcast_tracks.items() if len(track) > 0
    }
    tacticam_features = {
        tid: average_feature(track)
        for tid, track in tacticam_tracks.items() if len(track) > 0
    }

    ids_broadcast = list(broadcast_features.keys())
    ids_tacticam = list(tacticam_features.keys())

    feat_broadcast = np.array([broadcast_features[i] for i in ids_broadcast])
    feat_tacticam = np.array([tacticam_features[i] for i in ids_tacticam])

    if feat_broadcast.size == 0 or feat_tacticam.size == 0:
        print("No features to match.")
        return {}

    sim_matrix = cosine_similarity(feat_tacticam, feat_broadcast)

    # Mask out bad similarities
    sim_matrix[sim_matrix < 0.6] = -np.inf

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    mapping = {}
    used_broadcast_ids = set()
    for r, c in zip(row_ind, col_ind):
        sim_score = sim_matrix[r][c]
        if sim_score == -np.inf:
            continue
        tid_tacticam = ids_tacticam[r]
        tid_broadcast = ids_broadcast[c]
        if tid_broadcast in used_broadcast_ids:
            continue  # prevent duplicate mapping

        mapping[tid_tacticam] = tid_broadcast
        used_broadcast_ids.add(tid_broadcast)
        print(f"Mapping Tacticam ID {tid_tacticam} → Broadcast ID {tid_broadcast} [Similarity: {sim_score:.2f}]")

    return mapping
