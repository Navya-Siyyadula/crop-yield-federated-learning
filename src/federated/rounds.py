import random

# 🔹 Single round training
def run_round(global_model, clients):
    print("Running one round...")
    updates = []

    for i, c in enumerate(clients):
        print(f"{c} training...")

        # simulate training (random value)
        update = random.uniform(0.5, 1.5)
        updates.append(update)

    return updates


# 🔹 Full federated training (THIS WAS MISSING → ERROR FIX)
def run_federated_training(global_model, clients, num_rounds=3):
    print("🚀 Starting Federated Learning")

    for r in range(num_rounds):
        print(f"\n🔄 Round {r+1}")

        # Step 1: clients train
        updates = run_round(global_model, clients)

        # Step 2: aggregate updates (average)
        print("Aggregating updates...")
        avg_update = sum(updates) / len(updates)

        # Step 3: update global model
        print("Updating global model...")
        global_model.update(avg_update)

    print("\n✅ Training Complete")
    return global_model
