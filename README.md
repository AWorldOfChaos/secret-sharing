# Secret-sharing
This implementation of secret sharing was done as a part of the CS 406 course at IIT Bombay in Spring 2024.

## Usage
1. Run the following command to form shares for a given secret
python3 secret_sharing.py --secret path-to-secret-text --num_servers num-participants

2. Run the following command to decode the shares and obtain the secret (printed to terminal)
python3 retrieve_secret.py
