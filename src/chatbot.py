import os
import random
import pickle
import sys

def live_chat_loop():
    print("\n--- Initializing Intent Recognition Engine ---")
    
    # Load serializations
    try:
        with open('/tmp/artifacts/preprocessor.pkl', 'rb') as f: preprocessor = pickle.load(f)
        with open('/tmp/artifacts/model.pkl', 'rb') as f: model = pickle.load(f)
        with open('/tmp/artifacts/responses.pkl', 'rb') as f: responses = pickle.load(f)
    except FileNotFoundError:
        print("[CRITICAL] Trained model assets not found! Run training script first.")
        sys.exit(1)

    print("Engine Status: ACTIVE. Type 'exit' to terminate session.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() == 'exit':
            print("System shutting down. Goodbye.")
            break
            
        if not user_input.strip():
            continue

        # Process string stream through vector space
        query_vec = preprocessor.transform_query(user_input)
        tag, confidence = model.predict_intent(query_vec)

        # Anti-hallucination threshold fallback logic
        if confidence < 0.35:
            print("Bot: [Confidence Index Low] I'm sorry, I cannot verify the intent of that command.")
        else:
            reply = random.choice(responses[tag])
            print(f"Bot (Intent Match: {tag} | Conf: {confidence:.2f}): {reply}")

if __name__ == "__main__":
    # If model is not trained yet, auto-train before looping
    if not os.path.exists('/tmp/artifacts/model.pkl'):
        import train
        train.run_pipeline()
    live_chat_loop()