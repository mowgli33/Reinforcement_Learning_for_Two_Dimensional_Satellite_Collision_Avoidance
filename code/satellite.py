#class that describes the satelitte and its methods / attributes
class Satellite():
    def __init__(self, Vx0, f0):
        #Position
        self.Ox = 0
        self.Oy = 0

        #Vitesse
        self.Vx = Vx0
        self.Vy = 0

        #fuel pour la mission
        self.fuel = f0

        #Acceleration verticale
        self.ut = 0
    
    def get_satellite_position(self):
        pass


    def set_new_thrust(self):
        pass

    def update_position(self, dt):
        pass

    def update_fuel(self, dt):
        self.fuel -= 0.1 * dt * abs(self.ut)
    
    def is_out_of_fuel(self):
        pass

    def has_exceded_max_orbit(self, max_orbit):
        pass



