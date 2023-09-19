# SARS-CoV-2 Genome Sequence Prediction using LSTM

This project utilizes a Long Short-Term Memory (LSTM) network to predict nucleotide sequences in the SARS-CoV-2 genome. By training on actual genome sequences, the model aims to generate new, plausible sequences.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [How to Use](#how-to-use)
- [License](#license)
- [Author](#author)
- [Acknowledgements](#acknowledgements)

## Features

- **Data Retrieval**: Fetches SARS-CoV-2 complete genome sequences from the nucleotide database using the Entrez API.
- **LSTM Model**: Constructs and trains an LSTM neural network on the genome sequences to predict subsequent nucleotides.
- **Sequence Generation**: Uses the trained LSTM model to generate new nucleotide sequences.
- **Open Reading Frame (ORF) Detection**: Identifies ORFs in a provided viral genome sequence.

## Requirements

- Python 3.x
- BioPython
- Keras
- NumPy

## How to Use

1. Ensure you have all the required libraries installed.
2. Set your email in the `Entrez.email` variable for the Entrez query.
3. Run the script. It will fetch the genome sequences, train the LSTM model, generate new sequences, and identify ORFs in a sample viral genome.

## License

This project is licensed under the MIT License. For more details, refer to the accompanying license file.

## Author

- Emre Erdin

## Acknowledgements

- Genome sequences are sourced using the Entrez API from the nucleotide database.
- The LSTM model implementation is powered by Keras.
