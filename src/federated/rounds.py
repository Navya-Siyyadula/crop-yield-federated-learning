import random

def run_round(global_model, clients):
    print("Running one round...")
    updates = []

    for i, c in enumerate(clients):
        print(f"{c} training...")

        # simulate real training using random value
        update = random.uniform(0.5, 1.5)
        updates.append(update)

    return updates
