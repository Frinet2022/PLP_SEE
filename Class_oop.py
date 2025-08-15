class Literature:
    """Base class for literary works"""
    def __init__(self, title, author, year):
        self.title = title
        self.author = author
        self.year = year
        self.__secret_meaning = "Hidden symbolism" 
    
    def analyze(self):
        return f"Analyzing {self.title} by {self.author}"
    
    def reveal_secret(self):
        return self.__secret_meaning

class ShakespearePlay(Literature):
    """Shakespeare literature specialization"""
    def __init__(self, title, year, genre):
        super().__init__(title, "William Shakespeare", year)
        self.genre = genre
        self.famous_quote = "To be or not to be"
    
    def perform(self):
        return f"Performing {self.title} in iambic pentameter!"
    
    def analyze(self):  
        return f"Shakespearean analysis of {self.title}"

class Animal:
    """Base class for animals"""
    def __init__(self, name, habitat):
        self.name = name
        self.habitat = habitat
    
    def move(self):
        return "Generic animal movement"
    
    def communicate(self):
        return "Generic animal sound"

class Dolphin(Animal):
    """Dolphin specialization"""
    def __init__(self, name, pod_name):
        super().__init__(name, "Ocean")
        self.pod_name = pod_name
        self.intelligence = "High"
    
    def communicate(self):
        return "Eee-eee-eee clicks and whistles!"
    
    def jump(self):
        return f"{self.name} leaps gracefully out of the water!"

class Superhero:
    """Base class for superheroes"""
    def __init__(self, secret_identity, origin):
        self.__secret_identity = secret_identity  
        self.origin = origin
    
    def reveal_identity(self):
        return self.__secret_identity
    
    def use_power(self):
        return "Generic superhero power"

class Spiderman(Superhero):
    """Spiderman specialization"""
    def __init__(self):
        super().__init__("Peter Parker", "Radioactive spider bite")
        self.catchphrase = "With great power comes great responsibility"
    
    def use_power(self):
        return "Thwip! Shoots web from wrists"
    
    def swing(self):
        return "Swings between skyscrapers with web"
    





    class Vehicle:
        def move(self):
            pass
        class Car(Vehicle):
            def move(self):
                return "Driving 🚗"

class Plane(Vehicle):
    def move(self):
        return "Flying ✈️"

class Boat(Vehicle):
    def move(self):
        return "Sailing ⛵"

class Bicycle(Vehicle):
    def move(self):
        return "Pedaling 🚲"

# Polymorphism in action!
def travel(vehicle):
    print(vehicle.move())

# Create instances
vehicles = [Car(), Plane(), Boat(), Bicycle()]

# Test polymorphism
print("Let's go on a trip!")
for vehicle in vehicles:
    travel(vehicle)
