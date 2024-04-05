from utils import *
import random
import matplotlib.pyplot as plt

sizes = [10*(i+1) for i in range(100)]
y = []
for size in sizes:
    secrets = [32 for _ in range(size)]
    num_secrets = len(secrets)
    num_servers = 30

    num_serv,num_secr = padded_params(num_servers,num_secrets)
    num_padded = num_secr - num_secrets
    for i in range(num_padded):
        secrets.append(32)

    prime, omega2, omega3, order2, order3, shares = secret_share(secrets=secrets,num_servers=num_serv)
    y.append(len(shares)/(num_servers*num_secrets))

plt.plot(sizes, y, label='Space Efficiency')
plt.xlabel('Message Size')
plt.ylabel('Number of shares per secret')
plt.legend()
plt.savefig('Efficiency3.png')


