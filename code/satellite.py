import numpy as np


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

    def __init__(self, satellite: Satellite, num_debris=4):
        # Debris initialization
        self.num_debris = num_debris
        self.satellite = satellite
        self.debris_positions = self.initialize_debris()
        self.debris_velocities = self.initialize_debris_velocities()
        

    def initialize_debris(self):
        """Initialize debris positions randomly within a specified region."""
        return np.random.uniform(low=[1, -2], high=[5, 2], size=(self.num_debris, 2))

    def initialize_debris_velocities(self):
        """Initialize debris velocities based on random collision times."""
        collision_times = np.random.uniform(5, 10, self.num_debris)
        velocities = []
        for t_n in collision_times:
            Wx = (self.satellite.Sx + self.satellite.Vx * t_n - self.debris_positions[0][0]) / t_n
            Wy = (self.satellite.Sy + self.satellite.Vy * t_n - self.debris_positions[0][1]) / t_n
            velocities.append([Wx, Wy])
        return np.array(velocities)

    def update_debris(self, dt):
        """Update debris positions based on their velocities."""
        self.debris_positions += self.debris_velocities * dt

    def get_debris_states(self):
        """Return the current positions and velocities of all debris."""
        return self.debris_positions, self.debris_velocities


    def get_state(self):
        """Return the complete state including satellite and debris information."""
        debris_positions, debris_velocities = self.get_debris_states()
        return {
            'debris_positions': debris_positions,
            'debris_velocities': debris_velocities
        }

class Environment :
    def __init__(self, Vx0=0.5, f0=5, num_debris=4, max_orbit=2, beta=0.01, delta_t=0.1, max_time_steps=200):
        # Initialize the satellite
        self.satellite = Satellite(Vx0, f0)
        self.debris = Debris(satellite=self.satellite, num_debris=num_debris)
        self.max_orbit = max_orbit
        self.beta = beta
        self.delta_t = delta_t
        self.max_time_steps = max_time_steps
        self.time_step = 0


    def reset(self):
        """Reset the environment to its initial state."""
        self.satellite = Satellite(Vx0=0.5, f0=5)
        self.debris = Debris(self.satellite, num_debris=4)
        self.time_step = 0
        return self.get_state()
    
    def get_state(self):
        """Return the current state of the environment."""
        state = {**self.satellite.get_state(), **self.debris.get_state()}
        state['time_step'] = self.time_step
        return state
    
    def step(self, action):
        """Execute one time step within the environment."""
        # Update satellite and debris states
        self.satellite.ut = action  # Convert action to thrust level
        self.satellite.update_dynamics(self.delta_t, self.beta)
        self.satellite.update_fuel(self.delta_t)
        self.debris.update_debris(self.delta_t)
        self.time_step += 1

        # Calculate reward
        reward = self.calculate_reward()

        # Check for termination conditions
        done = self.check_termination()

        return self.get_state(), reward, done
    

    def calculate_collision_probability(self):
        """Calculate the probability of collision with each debris cluster."""
        distances = np.linalg.norm(self.debris.debris_positions - np.array([self.satellite.Sx, self.satellite.Sy]), axis=1)
        probabilities = np.where(distances <= 0.1, 0.005, 0.005 * np.exp(-(np.log(1000) / 4.9) * (distances - 0.1)))
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
    


# Define the action set
action_set = [-5,-3,-1,0,1,3,5]

# Example usage
env = Environment()
state = env.reset()
done = False

while not done:
    # Select a random action from the action set
    action = np.random.choice(action_set)
    state, reward, (done,termination_status) = env.step(action)
    print(f"State: {state}, Reward: {reward}, Done: {done}, Termination_cause: {termination_status}")



