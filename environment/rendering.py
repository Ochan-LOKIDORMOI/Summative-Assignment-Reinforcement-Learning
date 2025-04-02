import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import time
import random
from enum import Enum


class WildlifeRenderer:
    class State(Enum):
        NO_WILDLIFE = 0
        WILDLIFE_DISTANT = 1
        WILDLIFE_APPROACHING = 2
        WILDLIFE_IN_CROP = 3

    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.farm_area = [(1, 1), (1, 2), (2, 1), (2, 2)]  # Center 4 squares
        self.agent_pos = np.array([0, 0])
        self.wildlife_pos = []
        self.current_state = self.State.NO_WILDLIFE

        # Movement control
        self.last_agent_move = time.time()
        self.last_wildlife_move = time.time()
        self.agent_move_interval = 0.5
        self.wildlife_move_interval = 3.0

        # Episode tracking
        self.episode_count = 0
        self.max_episodes = 4
        self.episode_active = True
        self.simulation_active = True  # Controls all movement

        # Exploration tracking
        self.visited_cells = set()
        self.visited_cells.add(tuple(self.agent_pos))

        # Initialize GLUT
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(600, 600)
        self.window = glutCreateWindow(b"Wildlife Intrusion Prevention")

        glutDisplayFunc(self.render)
        glutIdleFunc(self.update)

        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, 1, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

        self.reset_episode()

    def reset_episode(self):
        """Reset environment for new episode"""
        if self.episode_count >= self.max_episodes:
            print("Maximum episodes reached - Simulation complete")
            self.simulation_active = False
            return

        self.episode_count += 1
        self.episode_active = True
        self.simulation_active = True

        # Reset positions
        self.agent_pos = np.array([0, 0])
        self.wildlife_pos = []
        self.visited_cells = set([tuple(self.agent_pos)])

        # Add 2-4 wildlife at random positions (not in farm)
        valid_positions = [(x, y) for x in range(self.grid_size)
                           for y in range(self.grid_size)
                           if (x, y) not in self.farm_area]

        for _ in range(random.randint(2, 4)):
            x, y = random.choice(valid_positions)
            self.wildlife_pos.append(np.array([x, y]))

        self.update_state()
        print(f"Starting episode {self.episode_count}/{self.max_episodes}")

    def move_wildlife_randomly(self):
        """Move each wildlife to a random adjacent cell"""
        if not self.simulation_active:
            return

        for i in range(len(self.wildlife_pos)):
            x, y = self.wildlife_pos[i]
            possible_moves = []

            if x > 0:
                possible_moves.append((-1, 0))
            if x < self.grid_size-1:
                possible_moves.append((1, 0))
            if y > 0:
                possible_moves.append((0, -1))
            if y < self.grid_size-1:
                possible_moves.append((0, 1))

            if possible_moves:
                dx, dy = random.choice(possible_moves)
                self.wildlife_pos[i] += np.array([dx, dy])

    def get_next_agent_position(self):
        """Find next position for systematic exploration"""
        unvisited = [(x, y) for x in range(self.grid_size)
                     for y in range(self.grid_size)
                     if (x, y) not in self.visited_cells]

        if unvisited:
            nearest = min(unvisited,
                          key=lambda p: np.linalg.norm(self.agent_pos - np.array(p)))
            direction = np.sign(np.array(nearest) - self.agent_pos)
            return self.agent_pos + direction
        else:
            return self.agent_pos + random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])

    def check_episode_conditions(self):
        """Check if episode conditions are met"""
        if not self.simulation_active:
            return True

        # For episodes 1-3: Complete when all animals are caught
        if self.episode_count < self.max_episodes:
            if len(self.wildlife_pos) == 0:
                print(
                    f"Episode {self.episode_count} complete: All animals caught!")
                self.episode_active = False
                return True

        # For final episode: Stop everything when any animal is outside farm
        else:
            if any(tuple(w) not in self.farm_area for w in self.wildlife_pos):
                print("Final episode: Simulation stopped (animal outside farm)")
                self.simulation_active = False
                return True

        return False

    def move_agent(self):
        if not self.simulation_active:
            return

        if time.time() - self.last_agent_move >= self.agent_move_interval:
            new_pos = np.clip(self.get_next_agent_position(),
                              0, self.grid_size-1)
            self.agent_pos = new_pos
            self.visited_cells.add(tuple(new_pos))

            # Check for wildlife collisions (capture)
            for i in range(len(self.wildlife_pos)-1, -1, -1):
                if np.array_equal(self.agent_pos, self.wildlife_pos[i]):
                    self.wildlife_pos.pop(i)

            self.last_agent_move = time.time()
            self.check_episode_conditions()

    def update_state(self):
        """Update wildlife state classification"""
        if not self.wildlife_pos:
            self.current_state = self.State.NO_WILDLIFE
            return

        # Calculate minimum distance of any wildlife to farm
        min_distance = min(
            min(np.linalg.norm(np.array(farm) - wildlife)
                for farm in self.farm_area)
            for wildlife in self.wildlife_pos
        )

        # Update state based on distances
        if any(tuple(w) in self.farm_area for w in self.wildlife_pos):
            self.current_state = self.State.WILDLIFE_IN_CROP
        elif min_distance <= 1:
            self.current_state = self.State.WILDLIFE_APPROACHING
        elif min_distance <= 2.5:
            self.current_state = self.State.WILDLIFE_DISTANT
        else:
            self.current_state = self.State.NO_WILDLIFE

    def update(self):
        if not self.simulation_active:
            glutPostRedisplay()
            return

        current_time = time.time()

        # Move wildlife every 3 seconds
        if current_time - self.last_wildlife_move >= self.wildlife_move_interval:
            self.move_wildlife_randomly()
            self.update_state()
            self.last_wildlife_move = current_time
            self.check_episode_conditions()

        # Move agent
        self.move_agent()

        # Start new episode if current one is complete
        if not self.episode_active and self.episode_count < self.max_episodes:
            time.sleep(1)
            self.reset_episode()

        glutPostRedisplay()

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(5, 10, 10, self.grid_size/2, self.grid_size/2, 0, 0, 0, 1)

        # Draw grid
        glBegin(GL_LINES)
        glColor3f(0.5, 0.5, 0.5)
        for x in range(self.grid_size + 1):
            glVertex3f(x, 0, 0)
            glVertex3f(x, self.grid_size, 0)
        for y in range(self.grid_size + 1):
            glVertex3f(0, y, 0)
            glVertex3f(self.grid_size, y, 0)
        glEnd()

        # Draw farm
        for x, y in self.farm_area:
            glPushMatrix()
            glTranslatef(x + 0.5, y + 0.5, 0)
            glBegin(GL_QUADS)
            glColor3f(0.2, 0.8, 0.2)
            glVertex3f(-0.5, -0.5, 0)
            glVertex3f(0.5, -0.5, 0)
            glVertex3f(0.5, 0.5, 0)
            glVertex3f(-0.5, 0.5, 0)
            glEnd()
            glPopMatrix()

        # Draw agent
        glPushMatrix()
        glTranslatef(self.agent_pos[0] + 0.5, self.agent_pos[1] + 0.5, 0)
        glBegin(GL_QUADS)
        glColor3f(0, 0, 1)
        glVertex3f(-0.4, -0.4, 0)
        glVertex3f(0.4, -0.4, 0)
        glVertex3f(0.4, 0.4, 0)
        glVertex3f(-0.4, 0.4, 0)
        glEnd()
        glPopMatrix()

        # Draw wildlife with state-based coloring
        for wildlife in self.wildlife_pos:
            glPushMatrix()
            glTranslatef(wildlife[0] + 0.5, wildlife[1] + 0.5, 0)

            # Set color based on state
            if self.current_state == self.State.WILDLIFE_IN_CROP:
                glColor3f(1.0, 0.0, 0.0)  # Red - in crop
            elif self.current_state == self.State.WILDLIFE_APPROACHING:
                glColor3f(1.0, 0.5, 0.0)  # Orange - approaching
            elif self.current_state == self.State.WILDLIFE_DISTANT:
                glColor3f(0.8, 0.8, 0.0)  # Yellow - distant
            else:
                glColor3f(0.8, 0.2, 0.2)  # Default red

            glBegin(GL_TRIANGLES)
            # Pyramid base
            glVertex3f(-0.4, -0.4, 0)
            glVertex3f(0.4, -0.4, 0)
            glVertex3f(0, 0.4, 0)
            # Pyramid sides
            glVertex3f(-0.4, -0.4, 0)
            glVertex3f(0, 0.4, 0)
            glVertex3f(0, 0, 0.6)
            glVertex3f(0.4, -0.4, 0)
            glVertex3f(0, 0.4, 0)
            glVertex3f(0, 0, 0.6)
            glVertex3f(-0.4, -0.4, 0)
            glVertex3f(0.4, -0.4, 0)
            glVertex3f(0, 0, 0.6)
            glEnd()
            glPopMatrix()

        # Draw status text
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, 600, 0, 600)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        status_text = f"Episode: {self.episode_count}/{self.max_episodes} | "
        status_text += f"Animals: {len(self.wildlife_pos)} | "
        status_text += f"State: {self.current_state.name.replace('_', ' ')}"

        if not self.simulation_active:
            status_text += " | Agent stopped"
        elif self.episode_count == self.max_episodes:
            status_text += " | FINAL EPISODE"

        glColor3f(1, 1, 1)
        glRasterPos2f(10, 580)
        for char in status_text:
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glutSwapBuffers()

    def run(self):
        print("Starting 4-episode simulation")
        print("Episodes 1-3: Complete when all animals caught")
        print("Final Episode: Everything stops when any animal is outside farm")
        print("Wildlife States:")
        print("- IN_CROP: Red")
        print("- APPROACHING: Orange")
        print("- DISTANT: Yellow")
        glutMainLoop()


if __name__ == "__main__":
    renderer = WildlifeRenderer()
    renderer.run()
