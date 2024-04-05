from utils import *
import matplotlib.pyplot as plt
import time

def shamir_numpy(prime=800_011,num_servers=3):
    """
    Uses numpy vectorisation for polynomial evaluation
    Works for fewer num_servers only due to overflow in numpy operations
    """
    secret = np.random.randint(low=1,high=prime)
    share_ids, shares = make_shamir_shares(secret,num_servers,num_servers,prime)
    reconstructed_secret = decode_shamir_shares(share_ids,shares,prime)
    
    assert secret == reconstructed_secret


def shamir(prime=800_011,num_servers=728):
    """
    Slower but works for secret sharing across more servers
    Uses Horner evaluation to prevent overflow in polynomial evaluation
    """
    secret = np.random.randint(low=1,high=prime)
    share_ids, shares = make_secure_shamir_shares(secret,num_servers,num_servers,prime)
    reconstructed_secret = decode_shamir_shares(share_ids,shares,prime)
    
    assert secret == reconstructed_secret

def fast_shamir(num_secrets):
    prime = 3173178172597642850885268374017
    order2 = 512
    order3 = 729

    # g = find_generator(prime) Very slow, so we precompute
    # order = prime - 1
    # omega2 = pow(g, order // order2, prime)
    # omega3 = pow(g, order // order3, prime)

    omega2 = 2640387605263602181035404433211
    omega3 = 1771403557660722120970917866078

    num_servers = order3 - 1
    num_coeffs = order2 - num_secrets - 1

    secrets = [random.randrange(prime) for _ in range(num_secrets)]
    shares = fast_shamir_share(secrets,num_servers=num_servers,num_coeffs=num_coeffs,omega2=omega2,omega3=omega3,prime=prime,order2=order2,order3=order3)
    reconstructed_secrets = decode_fast_shamir(shares=shares,num_secrets=num_secrets,omega2=omega2,omega3=omega3,order2=order2,order3=order3,prime=prime)

    assert secrets == reconstructed_secrets

if __name__ == "__main__":
    secret_array = [4*(i+1) for i in range(10)]
    time_ramp = []
    time_fast = []
    time_normal = []

    for num_secrets in secret_array:
        start_time = time.time()
        fast_shamir(num_secrets=num_secrets)
        end_time = time.time()
        execution_time = end_time - start_time
        time_ramp.append(execution_time)

    for num_secrets in secret_array:
        start_time = time.time()
        for j in range(num_secrets):
            fast_shamir(num_secrets=1)
        end_time = time.time()
        execution_time = end_time - start_time
        time_fast.append(execution_time)

    for num_secrets in secret_array:
        print(num_secrets)
        start_time = time.time()
        for j in range(num_secrets):
            shamir()
        end_time = time.time()
        execution_time = end_time - start_time
        time_normal.append(execution_time)

    plt.plot(secret_array, time_fast, label='fast_shamir')
    plt.plot(secret_array, time_normal, label='shamir')
    plt.plot(secret_array, time_ramp, label='ramp_sharing')
    plt.xlabel('Number of Secrets')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.savefig('Comparison4.png')
