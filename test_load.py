import pickle, traceback, sys

try:
    with open('modelfinal1.pkl', 'rb') as f:
        m = pickle.load(f)
    print("Model loaded OK — type:", type(m))
except Exception:
    traceback.print_exc()
    sys.exit(1)
