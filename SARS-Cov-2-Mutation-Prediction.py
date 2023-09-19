# -*- coding: utf-8 -*-
import random
import numpy as np
from Bio import SeqIO, Entrez
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

# Set the email for the Entrez query
Entrez.email = "emrerdin@gmail.com"

# Define the search query for retrieving the complete genomes of SARS-CoV-2
search_query = "SARS-CoV-2[Organism] AND complete genome"

# Use Entrez to search for the records
print("Searching for records...")
handle = Entrez.esearch(db="nucleotide", term=search_query, retmax=50)
search_results = Entrez.read(handle)
id_list = search_results["IdList"]

# Retrieve the nucleotide sequences for the retrieved records
print("Retrieving sequences...")
records = []
for i, record_id in enumerate(id_list):
    handle = Entrez.efetch(db="nucleotide", id=record_id, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    records.append(record)
    print(f"Retrieved sequence {i+1}/{len(id_list)}")

# Concatenate all the sequences into one long sequence
sequence = ''.join([str(record.seq) for record in records])

# Define the mapping between nucleotide characters and integers
chars = sorted(list(set(sequence)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Define the sequence length and step size for creating training examples
seq_length = 100
seq_step = 1

# Create the training examples by randomly sampling subsequences from the complete genome
print("Creating training examples...")
X = []
y = []
for i in range(0, len(sequence) - seq_length, seq_step):
    # Extract the input sequence and output sequence
    seq_in = sequence[i:i+seq_length]
    seq_out = sequence[i+seq_length]

    # Map the input sequence to a list of integers
    input_ints = [char_to_int[char] for char in seq_in]

    # Map the output sequence to a single integer
    output_int = char_to_int[seq_out]

    # Add the input sequence and output integer to the X and y lists, respectively
    X.append(input_ints)
    y.append(output_int)

print("X shape:", np.shape(X))
print("y shape:", np.shape(y))

# Reshape the training examples into a 3D array with shape (num_examples, seq_length, 1)
print("Reshaping training examples...")
print("X shape before:", np.shape(X))
X = np.reshape(X, (len(X), seq_length, 1))
print("X shape after:", np.shape(X))

# Normalize the training examples to have values between 0 and 1
print("Normalizing training examples...")
X = X / float(len(chars))

# One-hot encode the target values
print("One-hot encoding target values...")
print("y shape before:", np.shape(y))
y = np.array(y, dtype=int)
y = np.eye(len(chars))[y]
print("y shape after:", np.shape(y))

# Define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(len(chars), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define the early stopping callback to stop training if the validation loss does not improve for 10 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the model on the training examples with a validation split of 0.1 and batch size of 128
print("Training the model...")
model.fit(X, y, validation_split=0.1, batch_size=128, epochs=5, callbacks=[early_stop])

# Use the trained model to generate a new sequence of length 500
start_index = random.randint(0, len(sequence) - seq_length - 1)
seq = sequence[start_index:start_index + seq_length]
generated_seq = seq
for i in range(500):
    x = np.reshape([char_to_int[char] for char in seq], (1, seq_length, 1))
    x = x / float(len(chars))
    pred = model.predict(x, verbose=0)
    next_char = int_to_char[np.argmax(pred)]
    generated_seq += next_char
    seq = seq[1:] + next_char

print("Generated sequence:", generated_seq)

from keras.models import load_model
import random
import numpy as np

# Load the trained model
model = load_model('next-mutation-model.h5')

# Define the mapping between nucleotide characters and integers
chars = ['A', 'C', 'G', 'T']
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Define the length of the new sequence to be generated
seq_length = 100

# Choose a random starting point from the training data
start_index = random.randint(0, len(sequence) - seq_length - 1)
seq = sequence[start_index:start_index + seq_length]
generated_seq = seq

# Generate a new sequence of nucleotides using the trained model
for i in range(500):
    x = np.reshape([char_to_int[char] for char in seq], (1, seq_length, 1))
    x = x / float(len(chars))
    pred = model.predict(x, verbose=0)
    next_int = np.argmax(pred)
    if next_int >= len(chars):
        next_char = random.choice(chars)
    else:
        next_char = int_to_char[next_int]
    generated_seq += next_char
    seq = seq[1:] + next_char

print("Generated sequence:", generated_seq)

from Bio import Seq
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

viral_sequence = "GACCTACACAGGTGCCATCAAATTGGATGACAAAGATCCAAATTTCAAAGATCAAGTCATTTTGCTGAATAAGCATATTGACGCATACAAAACATTCCCATGCGGGAGACACGGGAACGGCAATACTTTAAACCCTTACAGGGTCGACACAAAGTGGGAGAGCAAGCCACCCGAACAGGGGGGCGTTCGTACGGGGCCCAGCGGTAGACCTAGAGAAGCAGCAGGCCACCCGAGAGGCAGGCGGGCAACCCCTACATGATGGGACCACCAGCGGGGCCTAAAAGAGGGCACGGCAGACATGCCTGGACCACGAGCAGGGGGGAAAGGACGCTAGAACAACAACGCACCACAGCGGGACAACAAGGGCCCCCCCGCCCAGCCAAGAAAGACCGGGGGGGGAGACGACCGACAGGGGGCGTTGGTGCACCGGGACAACCAAGCCTGACCACCAGGTTGACGTGCACGCCTACAAGCAAACACTCCACCGTACCAGAGCCAGCACGGTGGAAACAAAGCCGACGGCACGGAGACAAGGCGGCTCACTAGAGAAATCCGTATAACAACCAACACGCTCGCACCTGTGGAAAAAAAAGATGGCGA"

viral_genome = Seq(viral_sequence)

def find_orfs(sequence):
    orfs = []
    for frame in range(3):
        trans = sequence[frame:].translate(to_stop=True)
        start = trans.find('M')
        while start != -1:
            stop = trans.find('*', start)
            orfs.append(SeqRecord(Seq(trans[start:stop]), id=f"ORF_{frame}_{start}_{stop}"))
            start = trans.find('M', start + 1)
    return orfs

# Find ORFs in the given viral genome sequence
orfs = find_orfs(viral_genome)

# Print the ORFs
for orf in orfs:
    print(orf.id)
    print(orf.seq)