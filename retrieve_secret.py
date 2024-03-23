
from utils import *

with open("shares/params.txt", "r") as file:
    line = file.readline().strip()  # Read the first line and remove leading/trailing whitespaces
    prime, omega2, omega3, order2, order3, num_servers, num_secr = map(int, line.split(','))

shares = reconstruct_list(num_servers)
secrets = decode_fast_shamir(shares,num_secr,omega2,omega3,order2,order3,prime)
# secrets = secrets[:num_secrets]
secret = ''.join(chr(value) for value in secrets)
print(secret)