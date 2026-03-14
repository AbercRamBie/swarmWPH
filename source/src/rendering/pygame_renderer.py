"""
pygame_renderer.py — Real-time pygame visualization for swarm simulation.

Provides visual feedback during simulation execution with:
    - Predator/prey positions and states
    - Energy bars
    - Goal zone
    - Formation visualization
    - Statistics overlay
"""

import pygame
import math
from typing import List, Tuple, Optional
from ..core.states import PredatorMode


class PygameRenderer:
    """Real-time pygame renderer for swarm herding simulation."""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        fps: int = 60,
        show_stats: bool = True,
    ):
        """
        Initialize pygame renderer.

        Args:
            screen_width: Window width in pixels
            screen_height: Window height in pixels
            fps: Target frames per second
            show_stats: Whether to show statistics overlay
        """
        pygame.init()
        self.width = screen_width
        self.height = screen_height
        self.fps = fps
        self.show_stats = show_stats

        # Create display
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Energy-Aware Swarm Herding")

        # Store for dynamic title updates
        self.base_caption = "Energy-Aware Swarm Herding"

        # Clock for FPS control
        self.clock = pygame.time.Clock()

        # Colors
        self.BG_COLOR = (30, 30, 40)
        self.GOAL_COLOR = (80, 200, 120)
        self.GOAL_BORDER_COLOR = (60, 160, 100)

        self.PREDATOR_SEARCH_COLOR = (100, 100, 120)
        self.PREDATOR_PURSUE_COLOR = (220, 80, 80)
        self.PREDATOR_IDLE_COLOR = (60, 60, 80)

        self.PREY_COLOR = (255, 200, 100)
        self.PREY_DELIVERED_COLOR = (150, 150, 150)

        self.ENERGY_BAR_BG = (50, 50, 50)
        self.ENERGY_BAR_HIGH = (100, 220, 100)
        self.ENERGY_BAR_MED = (220, 200, 100)
        self.ENERGY_BAR_LOW = (220, 80, 80)

        self.TEXT_COLOR = (255, 255, 255)
        self.FORMATION_LINE_COLOR = (120, 120, 140)

        # Fonts (larger for better in-simulation readability)
        self.font_large = pygame.font.Font(None, 40)
        self.font_small = pygame.font.Font(None, 25)

        self.frame_count = 0

    def set_title(self, energy_model_name: str = None, config_name: str = None):
        """
        Update window title with energy model and configuration info.

        Args:
            energy_model_name: Name of the energy model being tested
            config_name: Configuration description (e.g., "Small: 5 pred, 10 prey")
        """
        title_parts = [self.base_caption]

        if energy_model_name:
            title_parts.append(f"[{energy_model_name}]")

        if config_name:
            title_parts.append(f"({config_name})")

        pygame.display.set_caption(" - ".join(title_parts))

    def handle_events(self) -> bool:
        """
        Process pygame events.

        Returns:
            False if quit event detected, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
        return True

    def render_frame(
        self,
        predators: List,
        prey_list: List,
        goal_zone: Tuple[float, float, float],
        charging_stations: Optional[List] = None,
        frame: int = 0,
        delivered_count: int = 0,
        total_energy: float = 0.0,
    ):
        """
        Render a single frame of the simulation.

        Args:
            predators: List of Predator objects
            prey_list: List of Prey objects
            goal_zone: (center_x, center_y, radius)
            charging_stations: Optional list of ChargingStation objects
            frame: Current frame number
            delivered_count: Number of prey delivered
            total_energy: Total energy consumed so far
        """
        self.screen.fill(self.BG_COLOR)
        self._draw_goal_zone(goal_zone)
        if charging_stations:
            for station in charging_stations:
                self._draw_charging_station(station)

        self._draw_formation_lines(predators, prey_list, goal_zone)

        for prey in prey_list:
            self._draw_prey(prey)

        for predator in predators:
            self._draw_predator(predator)

        if self.show_stats:
            self._draw_stats(frame, predators, delivered_count, total_energy, len(prey_list))
            self._draw_legend()

        pygame.display.flip()
        self.clock.tick(self.fps)
        self.frame_count += 1

    def _draw_goal_zone(self, goal_zone: Tuple[float, float, float]):
        """Draw the goal zone circle."""
        cx, cy, radius = goal_zone
        pygame.draw.circle(
            self.screen,
            self.GOAL_COLOR,
            (int(cx), int(cy)),
            int(radius),
        )
        pygame.draw.circle(
            self.screen,
            self.GOAL_BORDER_COLOR,
            (int(cx), int(cy)),
            int(radius),
            3,
        )

    def _draw_charging_station(self, station):
        """Draw a charging station."""
        x, y = station.position
        r = station.radius       
        pygame.draw.circle(
            self.screen,
            (255, 220, 100),
            (int(x), int(y)),
            int(r),
        )
        pygame.draw.circle(
            self.screen,
            (200, 180, 80),
            (int(x), int(y)),
            int(r),
            2,
        )

    def _draw_formation_lines(self, predators: List, prey_list: List, goal_zone: Tuple):       
        gx, gy, _ = goal_zone

        for predator in predators:
            if predator.mode != PredatorMode.PURSUE:
                continue
            if predator.assigned_prey_index is None:
                continue
            if predator.assigned_prey_index >= len(prey_list):
                continue

            prey = prey_list[predator.assigned_prey_index]
            if prey.delivered:
                continue

            # Draw line from predator to prey
            px, py = predator.position
            prey_x, prey_y = prey.position

            pygame.draw.line(
                self.screen,
                self.FORMATION_LINE_COLOR,
                (int(px), int(py)),
                (int(prey_x), int(prey_y)),
                1,
            )

    def _draw_prey(self, prey):
        x, y = prey.position
        r = prey.radius

        if prey.delivered:
            color = self.PREY_DELIVERED_COLOR
        else:
            color = self.PREY_COLOR

        pygame.draw.circle(self.screen, color, (int(x), int(y)), int(r))
        pygame.draw.circle(self.screen, (200, 150, 80), (int(x), int(y)), int(r), 2)

    def _draw_predator(self, predator):
        """Draw a predator agent with energy bar and directional indicator."""
        x, y = predator.position
        r = predator.radius
        if predator.mode == PredatorMode.IDLE:
            color = self.PREDATOR_IDLE_COLOR
        elif predator.mode == PredatorMode.SEARCH:
            color = self.PREDATOR_SEARCH_COLOR
        else: 
            color = self.PREDATOR_PURSUE_COLOR
        pygame.draw.circle(self.screen, color, (int(x), int(y)), int(r))
        pygame.draw.circle(self.screen, (180, 60, 60), (int(x), int(y)), int(r), 2)
        heading = predator.heading_radians
        tip_x = x + r * 1.5 * math.cos(heading)
        tip_y = y + r * 1.5 * math.sin(heading)
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (int(x), int(y)),
            (int(tip_x), int(tip_y)),
            2,
        )

        self._draw_energy_bar(predator, x, y - r - 10)

    def _draw_energy_bar(self, predator, x: float, y: float):
        """Draw energy bar above predator."""
        bar_width = 30
        bar_height = 4
        pygame.draw.rect(
            self.screen,
            self.ENERGY_BAR_BG,
            (int(x - bar_width // 2), int(y), bar_width, bar_height),
        )

        energy_fraction = predator.energy_remaining / predator.energy_capacity
        fill_width = int(bar_width * energy_fraction)

        if energy_fraction > 0.6:
            color = self.ENERGY_BAR_HIGH
        elif energy_fraction > 0.3:
            color = self.ENERGY_BAR_MED
        else:
            color = self.ENERGY_BAR_LOW

        if fill_width > 0:
            pygame.draw.rect(
                self.screen,
                color,
                (int(x - bar_width // 2), int(y), fill_width, bar_height),
            )

    def _draw_stats(
        self,
        frame: int,
        predators: List,
        delivered_count: int,
        total_energy: float,
        total_prey: int,
    ):
        """Draw statistics overlay in top-left corner."""
        stats_x = 10
        stats_y = 10
        line_height = 34
        active_count = sum(1 for p in predators if p.mode == PredatorMode.PURSUE)

        stats_lines = [
            f"Frame: {frame}",
            f"Delivered: {delivered_count}/{total_prey}",
            f"Active Predators: {active_count}/{len(predators)}",
            f"Total Energy: {total_energy:.1f}",
        ]

        for i, line in enumerate(stats_lines):
            text_surface = self.font_small.render(line, True, self.TEXT_COLOR)
            self.screen.blit(text_surface, (stats_x, stats_y + i * line_height))

    def _draw_legend(self):
        """Draw compact legend below stats in top-left corner."""
        panel_width = 300
        panel_height = 150
        panel_x = 10       
        line_height = 34
        stats_line_count = 4
        panel_y = 10 + stats_line_count * line_height + 8

        pygame.draw.rect(
            self.screen,
            (20, 20, 30),
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=6,
        )
        pygame.draw.rect(
            self.screen,
            (90, 90, 90),
            (panel_x, panel_y, panel_width, panel_height),
            1,
            border_radius=6,
        )

        entries = [
            ("Herder - Active", self.PREDATOR_PURSUE_COLOR),
            ("Herder - Searching/Conserving", self.PREDATOR_SEARCH_COLOR),
            ("Target - Active", self.PREY_COLOR),
            ("Target - Delivered", self.PREY_DELIVERED_COLOR),
        ]

        base_y = panel_y + 24
        row_gap = 34
        marker_x = panel_x + 18
        text_x = panel_x + 34

        for index, (label, color) in enumerate(entries):
            cy = base_y + index * row_gap
            pygame.draw.circle(self.screen, color, (marker_x, cy), 7)
            text_surface = self.font_small.render(label, True, self.TEXT_COLOR)
            self.screen.blit(text_surface, (text_x, cy - 12))

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()
