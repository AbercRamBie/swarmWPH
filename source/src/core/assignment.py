"""
assignment.py — Decentralized prey assignment with conflict resolution.

Each predator independently selects the closest undelivered prey.
When multiple predators select the same prey, they resolve conflicts
through local negotiation over multiple rounds. The predator with the
highest claim strength wins.

Claim strength = 0.7 * energy_remaining - 0.3 * distance_to_prey

Bug fixes applied:
    - ✓ Bug 5: claim_strength evaluates candidate k instead of winner

References:
    - Decentralized coordination algorithm
    - Market-based task allocation
"""

from typing import List, Dict, Set
from src.core.predator import Predator
from src.core.prey import Prey
from src.utils.math_helpers import distance


def assign_prey_to_predators(
    predators: List[Predator],
    prey_list: List[Prey],
    comm_radius: float,
    conflict_rounds: int,
) -> int:
    """
    Decentralized prey assignment with conflict resolution.

    Process:
        1. Each predator picks closest undelivered prey
        2. Neighbors within comm_radius exchange choices
        3. Conflicts resolved over multiple rounds using claim strength
        4. Losers pick next-best prey (excluding neighbor claims)

    Args:
        predators: List of all predator agents
        prey_list: List of all prey agents
        comm_radius: Communication range for neighbor detection
        conflict_rounds: Number of rounds to resolve conflicts

    Returns:
        Total number of communication messages sent
    """
    message_count = 0

    # Step 1: Each predator picks closest undelivered prey
    for pred in predators:
        if pred.disengaged or pred.energy_remaining <= 0:
            pred.assigned_prey_index = None
            continue

        closest_idx = None
        closest_dist = float('inf')

        for i, prey in enumerate(prey_list):
            if prey.delivered:
                continue

            dist = distance(pred.position, prey.position)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i

        pred.assigned_prey_index = closest_idx
        pred.formation_slot_index = 0  # Will be set during conflict resolution

    # Step 2: Build neighbor graph (who is within comm range of whom)
    neighbors: Dict[int, Set[int]] = {i: set() for i in range(len(predators))}

    for i, pred_i in enumerate(predators):
        for j, pred_j in enumerate(predators):
            if i >= j:
                continue  # Avoid duplicates and self

            dist = distance(pred_i.position, pred_j.position)
            if dist <= comm_radius:
                neighbors[i].add(j)
                neighbors[j].add(i)
                message_count += 2  # Bidirectional discovery

    # Step 3: Conflict resolution over multiple rounds
    for round_num in range(conflict_rounds):
        message_count += _resolve_conflicts_one_round(
            predators, prey_list, neighbors
        )

    return message_count


def _resolve_conflicts_one_round(
    predators: List[Predator],
    prey_list: List[Prey],
    neighbors: Dict[int, Set[int]],
) -> int:
    """
    One round of conflict resolution.

    For each predator:
        - Check if any neighbor wants the same prey
        - If conflict, compare claim strengths
        - Loser picks next-best prey (excluding neighbor claims)

    Bug fix #5: Evaluate claim strength of candidate k, not winner!

    Args:
        predators: List of all predator agents
        prey_list: List of all prey agents
        neighbors: Neighbor graph (predator_id -> set of neighbor ids)

    Returns:
        Number of messages exchanged this round
    """
    message_count = 0

    for i, pred in enumerate(predators):
        if pred.assigned_prey_index is None:
            continue

        my_prey = pred.assigned_prey_index

        # Find neighbors who want the same prey (contest)
        contest = [i]  # Start with self
        for neighbor_id in neighbors[i]:
            neighbor = predators[neighbor_id]
            if neighbor.assigned_prey_index == my_prey:
                contest.append(neighbor_id)
                message_count += 1  # Exchange choice

        if len(contest) == 1:
            # No conflict
            pred.formation_slot_index = 0
            continue

        # Conflict! Determine winner by claim strength
        # Bug fix #5: Was evaluating winner repeatedly, now correctly evaluates each candidate
        winner = contest[0]
        winner_strength = predators[winner].get_claim_strength(prey_list[my_prey].position)

        for k in contest[1:]:
            # Bug fix #5: Was "claim_strength(predators[winner], ...)", now correctly:
            candidate_strength = predators[k].get_claim_strength(prey_list[my_prey].position)
            message_count += 1  # Strength comparison message

            if candidate_strength > winner_strength:
                winner = k
                winner_strength = candidate_strength

        # If this predator lost, pick next-best prey
        if winner != i:
            # Collect all prey claimed by neighbors
            neighbor_claims = set()
            for neighbor_id in neighbors[i]:
                neighbor_prey = predators[neighbor_id].assigned_prey_index
                if neighbor_prey is not None:
                    neighbor_claims.add(neighbor_prey)

            # Pick closest prey NOT claimed by neighbors
            closest_idx = None
            closest_dist = float('inf')

            for j, prey in enumerate(prey_list):
                if prey.delivered or j in neighbor_claims:
                    continue

                dist = distance(pred.position, prey.position)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = j

            pred.assigned_prey_index = closest_idx
            pred.formation_slot_index = 0

        # If winner, assign formation slot
        else:
            # Count how many in contest (determines formation slot)
            pred.formation_slot_index = contest.index(i)

    return message_count
