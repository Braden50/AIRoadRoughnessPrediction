import random

rate = 0.25
num_data = 178
portions = 20
catch = num_data / portions
# 5 mph
print(f"Elapsed: {num_data * rate} seconds")
modes = [50, 122, 183, 200, 88, 230, 20, 130]
constant_modes = [10, 70, 30, 50, 130, -5]
sigmas = [10, 20, 6, 5, 15]


maintain = False
last = random.choice(modes)
with open('data.csv', 'w') as f:
    for i in range(num_data):
        sigma = random.choice(sigmas) * 2
        if i % (num_data / portions) == 0:
            if maintain: # already maintaining
                maintain = random.choice([True, True, True, False])
            else:  # not maintaining
                maintain = random.choice([True, False, False, False, False, False])
        if maintain:
            mode = random.choice(modes)
        else:
            mode = random.choice(constant_modes)
        
        x = random.gauss(mode, sigma)
        if x < 0:
            x = 0
        if x > 255:
            x = 255
        f.write(f"{x},\n")