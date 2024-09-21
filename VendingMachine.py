import psycopg2 as psql

class Condiment:
    def __init__(self, num_sugar, num_milk):
        self.sugar = num_sugar
        self.milk = num_milk

    def get_condiment_price(self):
        sugar_cost = self.sugar * 0.10
        milk_cost = self.milk * 0.15
        return sugar_cost + milk_cost

class Beverage:
    def __init__(self, name, base_price, milk_allowed=False):
        self.name = name
        self.base_price = base_price
        self.milk_allowed = milk_allowed
        self.condiment_price = 0

    def add_condiments(self, num_sugar, num_milk):
        is_ordering = True
        while is_ordering:
            num_condiments = num_sugar + num_milk
            if num_condiments <= 3:
                if not self.milk_allowed and num_milk > 0:
                    print("Milk is not allowed for this beverage.")
                    continue
                self.condiment_price = Condiment(num_sugar, num_milk).get_condiment_price()
                break
            else:
                print("Please choose at most 3 condiments.")
                response = input("Would you like to change your condiments (yes/no)? ")
                if response.lower() == "no":
                    is_ordering = False
                num_sugar = int(input("How much sugar would you like to add? "))
                num_milk = int(input("How much milk would you like to add? "))

    def get_price(self):
        return self.base_price + self.condiment_price

    def __str__(self):
        if self.name == "Regular Coffee":
            return f'{self.name}: $1.10'
        elif self.name == "Espresso":
            return f'{self.name}: $2.00'
        else:
            return f'{self.name}: $3.15'

class VendingMachine:
    def __init__(self):
        self.beverages = {
            '1': Beverage("Regular Coffee", 1.10, milk_allowed=True),
            '2': Beverage("Espresso", 2.00, milk_allowed=True),
            '3': Beverage("Cappuccino", 3.15, milk_allowed=False)
        }

def create_table():
    try:
        conn = psql.connect(user="postgres",
                            password="",
                            host="localhost",
                            port="5432",
                            database="VendingMachine")
    except Exception as e:
        print(e)

    cur = conn.cursor()
    cur.execute("""CREATE TABLE VendingMachineOrders(
                order_id SERIAL PRIMARY KEY,
                beverage VARCHAR NOT NULL,
                condiments VARCHAR NOT NULL,
                price VARCHAR NOT NULL);
                """)
    conn.commit()
    cur.close()
    conn.close()
def main():
    print("Welcome to the Coffee Machine!")
    print("1. Regular Coffee ($1.10)")
    print("2. Espresso ($2.00)")
    print("3. Cappuccino ($3.15)")

    choice = input("Please select a beverage(1-3): ")
    beverage = VendingMachine().beverages[choice]
    print(f'You have chosen {beverage.name}')

    response = input("Would you like to add any sugar or milk (yes/no)? You can only add a maximum of 3 condiments. ")
    milk_units = 0
    sugar_units = 0
    if response.lower() == "yes":
        sugar_units = int(input("How much sugar would you like (0-3)? "))
        milk_units = int(input("How much milk would you like to add (0-3)? "))
    beverage.add_condiments(sugar_units, milk_units)

    print(f'Your total is: ${beverage.get_price():.2f}')

# Main execution
if __name__ == "__main__":
    main()
