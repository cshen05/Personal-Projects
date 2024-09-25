from abc import ABC, abstractmethod

class Dog(ABC):
    def __init__(self, name, breed, height):
        self.name = name
        self.breed = breed
        self.height = height
        
    @abstractmethod
    def print_details(self):
        pass
    
    @abstractmethod
    def fur_type(self):
        pass
    
    def walk(self):
        print("I walk on 4 legs")
        
class Poodle(Dog):
    def print_details(self):
        print("name: ", self.name, " breed: ", self.breed, " height: ", self.height)
        
    def fur_type(self):
        print("I have curly fur")
        
class GreatDane(Dog):
    def print_details(self):
        print("I am a dog with ame: ", self.name, " breed: ", self.breed, " height: ", self.height)
        
    #def fur_type(self):
        #print("I have short fur")