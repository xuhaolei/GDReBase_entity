import pickle

with open('vocab.pkl','rb') as f:
	vocab = pickle.load(f)
print(vocab)