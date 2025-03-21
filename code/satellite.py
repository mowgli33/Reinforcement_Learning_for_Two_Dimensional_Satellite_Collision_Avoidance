import numpy as np
import pygame


#class that describes the satelitte and its methods / attributes
class Satellite:
    def __init__(self, Vx0, f0):
        #Position
        self.Sx = 0
        self.Sy = 0

        #Vitesse
        self.Vx = Vx0
        self.Vy = 0

        #fuel pour la mission
        self.f = f0

        #Acceleration verticale
        self.ut = 0
 
    def get_satellite_position(self):
        return (self.Sx, self.Sy)

    def update_dynamics(self, dt, beta):
        # Calculate the next vertical velocity
        Vy_next = self.Vy + self.ut * dt

        # Calculate the Gaussian error term epsilon
        variance = beta**2 * abs(self.Vy + Vy_next)**2
        std_dev = np.sqrt(variance)
        epsilon = np.random.normal(0, std_dev)

        # Update the satellite's position
        self.Sx += self.Vx * dt
        self.Sy += (self.Vy + Vy_next) / 2 * dt + epsilon

        # Update the satellite's velocity
        self.Vy += self.ut*dt

    def update_fuel(self, dt):
        self.f -= 0.1 * dt * abs(self.ut)
    
    def is_out_of_fuel(self):
        return self.f <= 0

    def has_exceded_max_orbit(self, max_orbit):
        return abs(self.Sy) > max_orbit
    
    def get_state(self):
        """Return the complete state including satellite and debris information."""
        return {
            'satellite_position': (self.Sx, self.Sy),
            'satellite_velocity': (self.Vx, self.Vy),
            'fuel': self.f
        }


class Debris :

    def __init__(self, id, satellite: Satellite, dt):
        # Debris initialization
        self.Sx, self.Sy = np.random.uniform(low=[1, -2], high=[5, 2])
        self.Wx, self.Wy = self.initialize_debris_velocities(satellite, dt)
        self.id = id
        

    def initialize_debris_velocities(self, satellite: Satellite, dt):
        """Initialize debris velocities based on random collision times."""
        collision_times = np.random.randint(50, 100)*dt
        Wx = (satellite.Sx + satellite.Vx * collision_times - self.Sx) / collision_times
        Wy = (satellite.Sy + satellite.Vy * collision_times - self.Sy) / collision_times
        return (Wx,Wy)

    def update_debris(self, dt):
        """Update debris positions based on their velocities."""
        self.Sx += self.Wx * dt
        self.Sy += self.Wy * dt


    def get_state(self):
        """Return the complete state including satellite and debris information."""
        return {
            'debris_positions': (self.Sx, self.Sy),
            'debris_velocities': (self.Wx, self.Wy)
        }
    
class Set_debris :

    def __init__(self,num_debris, satellite, delta_t):
        self.num_debris = num_debris
        self.delta_t = delta_t

        set_debris = set()
        id = 0
        for n in range(self.num_debris) :
            set_debris.add(Debris(id = id, satellite=satellite, dt = self.delta_t))
            id+=1

        self.set_debris = set_debris

    def get_state(self):

        Dict = {}
        for d in self.set_debris :
            Dict[d.id] = d.get_state()
        return(Dict)
    
    def update_debris(self, delta_t):
        for d in self.set_debris :
            d.update_debris(delta_t)




class Environment :
    def __init__(self, Vx0=0.5, f0=5, num_debris=4, max_orbit=2, beta=0.01, delta_t=0.1, max_time_steps=200):
        # Initialize the satellite
        self.satellite = Satellite(Vx0, f0)
        self.num_debris = num_debris
        self.max_orbit = max_orbit
        self.beta = beta
        self.delta_t = delta_t
        self.max_time_steps = max_time_steps
        self.time_step = 0
        self.Set_debris = Set_debris(self.num_debris, self.satellite, self.delta_t)

    def reset(self):
        """Reset the environment to its initial state."""
        self.satellite = Satellite(Vx0=0.5, f0=5)
        self.Set_debris = Set_debris(self.num_debris, self.satellite, self.delta_t)
        self.time_step = 0
        return self.get_state()
    
    def get_state(self):
        """Return the current state of the environment."""
        state = {**self.satellite.get_state(), **self.Set_debris.get_state()}
        state['time_step'] = self.time_step
        return state
    
    def step(self, action):
        """Execute one time step within the environment."""
        # Update satellite and debris states
        self.satellite.ut = action  # Convert action to thrust level
        self.satellite.update_dynamics(self.delta_t, self.beta)
        self.satellite.update_fuel(self.delta_t)
        self.Set_debris.update_debris(self.delta_t)
        self.time_step += 1

        # Calculate reward
        reward = self.calculate_reward()

        # Check for termination conditions
        done = self.check_termination()

        return self.get_state(), reward, done
    

    def calculate_collision_probability(self):
        """Calculate the probability of collision with each debris cluster."""
        distances = []
        for d in self.Set_debris.set_debris :
            distances.append(np.linalg.norm(np.array([d.Sx, d.Sy]) - np.array([self.satellite.Sx, self.satellite.Sy])))
        
        probabilities = np.where(np.array(distances) <= 0.1, 0.005, 0.005 * np.exp(-(np.log(1000) / 4.9) * (np.array(distances) - 0.1)))
        return probabilities

    def calculate_reward(self):
        """Calculate the reward based on the current state."""
        Sx, Sy = self.satellite.get_satellite_position()
        fuel_penalty = -0.1 * self.satellite.ut if self.satellite.ut != 0 else 0
        deviation_penalty = -abs(Sy) / self.max_orbit
        collision_probabilities = self.calculate_collision_probability()
        collision_penalty = -100*np.sum(collision_probabilities)

        reward = 5 + fuel_penalty + deviation_penalty + collision_penalty
        return reward

    def check_termination(self):
        """Check if the episode should terminate and return the reason."""
        if self.satellite.is_out_of_fuel():
            return True, "Out of fuel"
        if self.satellite.has_exceded_max_orbit(self.max_orbit):
            return True, "Exceeded maximum orbit"
        if self.time_step >= self.max_time_steps:
            return True, "Reached maximum time steps"
        # Check for collision with debris using collision probabilities
        collision_probabilities = self.calculate_collision_probability()
        collision_occurred = np.random.rand() < np.max(collision_probabilities)
        if collision_occurred:
            return True, "Collision with debris"
        # If no termination condition is met
        return False, ""
    

    def draw(self, screen, width, height):
        # Clear screen
        screen.fill(black)

        # Draw satellite
        sat_pos = self.satellite.get_satellite_position()
        sat_x = int((sat_pos[0] / 20) * width)  # Scale to screen width
        sat_y = int((1 - (sat_pos[1] + 2) / 4) * height)  # Scale to screen height
        pygame.draw.circle(screen, blue, (sat_x, sat_y), 10)

        # Draw debris
        for debris in self.Set_debris.set_debris:
            debris_pos = debris.get_state()['debris_positions']
            debris_x = int((debris_pos[0] / 20) * width)  # Scale to screen width
            debris_y = int((1 - (debris_pos[1] + 2) / 4) * height)  # Scale to screen height
            pygame.draw.circle(screen, red, (debris_x, debris_y), 5)

        # Update display
        pygame.display.flip()


# Define the action set
action_set = [-5,-3,-1,0,1,3,5]

# Example usage
env = Environment()
state = env.reset()
done = False

# while not done:
#     # Select a random action from the action set
#     action = np.random.choice(action_set)
#     state, reward, (done,termination_status) = env.step(action)
#     print(f"State: {state}, Reward: {reward}, Done: {done}, Termination_cause: {termination_status}")




# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Satellite Collision Avoidance")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)


# Main loop
clock = pygame.time.Clock()
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Select a random action from the action set
    action = np.random.choice(action_set)
    state, reward, (done, termination_status) = env.step(action)

    # Draw the environment
    env.draw(screen, width, height)

    # Cap the frame rate
    clock.tick(30)

    # Print status
    print(f"State: {state}, Reward: {reward}, Done: {done}, Termination Cause: {termination_status}")

pygame.quit()
