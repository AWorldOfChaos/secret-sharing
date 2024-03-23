from utils import *
import argparse
import random
    
parser = argparse.ArgumentParser(description="Read contents of a file into a string")
parser.add_argument("--secret", type=str, help="Path to the file containing the secret")
parser.add_argument("--num_servers", type=int, help="Number of servers")
args = parser.parse_args()

file_contents = read_file_contents(args.secret)
if file_contents:
    secrets = string_to_ascii_list(file_contents)
    num_secrets = len(secrets)
    num_servers = args.num_servers

    num_serv,num_secr = padded_params(num_servers,num_secrets)
    num_padded = num_secr - num_secrets
    for i in range(num_padded):
        secrets.append(32)
   
    prime, omega2, omega3, order2, order3, shares = secret_share(secrets=secrets,num_servers=num_serv)
    split = split_list_into_k_parts(shares, num_servers)

    for i in range(len(split)):
        lst = split[i]
        with open(f"shares/share{i}.txt", "w") as file:
            file.write(f"{i}:{lst}")

    with open(f"shares/params.txt", "w") as file:
        file.write(f"{prime}, {omega2}, {omega3}, {order2}, {order3}, {num_servers}, {num_secr}")

    
