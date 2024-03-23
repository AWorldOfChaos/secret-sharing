import numpy as np
import random
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

def string_to_ascii_list(input_string):
    return [ord(char) for char in input_string]

def read_file_contents(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    
def power_of_3_greater_than_x(x):
    power = 1
    while 3 ** power <= x:
        power += 1
    return 3 ** power

def power_of_2_greater_than_x(x):
    power = 1
    while 2 ** power <= x:
        power += 1
    return 2 ** power

def add_mod_p(a, b, p=800_011):
    result = (a + b) % p
    return result

def multiply_mod_p(a, b, p=800_011):
    result = (a * b) % p
    return result

def subtract_mod_p(a, b, p=800_011):
    result = (a - b) % p
    return result

def _extended_gcd(a, b):
    """
    Division in integers modulus p means finding the inverse of the
    denominator modulo p and then multiplying the numerator by this
    inverse (Note: inverse of A is B such that A*B % p == 1). This can
    be computed via the extended Euclidean algorithm
    http://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Computation
    """
    x = 0
    last_x = 1
    y = 1
    last_y = 0
    while b != 0:
        quot = a // b
        a, b = b, a % b
        x, last_x = last_x - quot * x, x
        y, last_y = last_y - quot * y, y
    return last_x, last_y

def divide_mod_p(num, den, p=800_011):
    inv, _ = _extended_gcd(den, p)
    return (num * inv) % p

def inverse(a, prime):
    inv, _ = _extended_gcd(a, prime)
    return inv

def evaluate_polynomial(coefficients, x):
    result = np.polyval(coefficients, x)
    return result

def make_shamir_shares(secret, minimum=5,num_servers=5,prime = 800_011):
    seed = 0
    np.random.seed(seed=seed)

    coefficients = np.random.randint(low=1,high=prime,size=minimum-1)
    coefficients = np.append(coefficients, secret)
    share_ids = np.random.randint(low=1,high=prime,size=num_servers)
    while len(share_ids) != len(set(share_ids)):
        share_ids = np.random.randint(low=1,high=prime,size=num_servers)
    shares = np.mod(evaluate_polynomial(coefficients,share_ids), prime)

    return share_ids, shares

def evaluate_overflow_polynomial(coefficients, x, prime = 800_011):
    results = []
    if type(x) == list:
        for i in x:
            val = 0
            for coeff in coefficients:
                val = (val*x + coeff) % prime 
            results.append(val)
    else:
        val = 0
        for coeff in coefficients:
            val = (val*x + coeff) % prime 
        results = val
  
    return np.array(results)

def make_secure_shamir_shares(secret, minimum=5,num_servers=5,prime = 800_011):
    seed = 0
    np.random.seed(seed=seed)

    coefficients = np.random.randint(low=1,high=prime,size=minimum-1)
    coefficients = np.append(coefficients, secret)
    share_ids = np.random.randint(low=1,high=prime,size=num_servers)
    while len(share_ids) != len(set(share_ids)):
        share_ids = np.random.randint(low=1,high=prime,size=num_servers)
    shares = np.mod(evaluate_overflow_polynomial(coefficients,share_ids, prime), prime)

    return share_ids, shares

def decode_shamir_shares(share_ids, shares,prime = 800_011,value=0):
    assert len(share_ids) == len(shares)
    assert len(share_ids) == len(set(share_ids))
    secret = 0
    for i in range(len(share_ids)):
        total = 1
        for j in range(len(share_ids)):
            if i != j:
                total = multiply_mod_p(total, 
                                       divide_mod_p(subtract_mod_p(value,share_ids[j],prime),
                                                    subtract_mod_p(share_ids[i],share_ids[j],prime),
                                                    prime),
                                       prime) 
        
        total = multiply_mod_p(shares[i],total,prime)
        secret = add_mod_p(secret,total,prime)

    return secret

def fft2_forward(aX, omega,prime):
    if len(aX) == 1:
        return aX

    # split A(x) into B(x) and C(x) -- A(x) = B(x^2) + x C(x^2)
    bX = aX[0::2]
    cX = aX[1::2]
    
    # apply recursively
    omega_squared = pow(omega, 2, prime)
    B = fft2_forward(bX, omega_squared,prime)
    C = fft2_forward(cX, omega_squared,prime)
        
    # combine subresults
    A = [0] * len(aX)
    Nhalf = len(aX) >> 1
    point = 1
    for i in range(0, Nhalf):
        
        x = point
        A[i]         = (B[i] + x * C[i]) % prime
        A[i + Nhalf] = (B[i] - x * C[i]) % prime

        point = (point * omega) % prime
        
    return A

def fft2_backward(A,omega,prime):
    N_inv = inverse(len(A),prime)
    return [ (a * N_inv) % prime for a in fft2_forward(A, inverse(omega,prime), prime) ]

def fft3_forward(aX, omega, prime):
    if len(aX) <= 1:
        return aX

    # split A(x) into B(x), C(x), and D(x): A(x) = B(x^3) + x C(x^3) + x^2 D(x^3)
    bX = aX[0::3]
    cX = aX[1::3]
    dX = aX[2::3]
    
    # apply recursively
    omega_cubed = pow(omega, 3, prime)
    B = fft3_forward(bX, omega_cubed,prime)
    C = fft3_forward(cX, omega_cubed,prime)
    D = fft3_forward(dX, omega_cubed,prime)
        
    # combine subresults
    A = [0] * len(aX)
    Nthird = len(aX) // 3
    omega_Nthird = pow(omega, Nthird, prime)
    point = 1
    for i in range(Nthird):
        
        x = point
        xx = (x * x) % prime
        A[i] = (B[i] + x * C[i] + xx * D[i]) % prime
        
        x = x * omega_Nthird % prime
        xx = (x * x) % prime
        A[i + Nthird] = (B[i] + x * C[i] + xx * D[i]) % prime
        
        x = x * omega_Nthird % prime
        xx = (x * x) % prime
        A[i + Nthird + Nthird] = (B[i] + x * C[i] + xx * D[i]) % prime

        point = (point * omega) % prime
        
    return A

def fft3_backward(A,omega,prime):
    N_inv = inverse(len(A),prime)
    return [ (a * N_inv) % prime for a in fft3_forward(A, inverse(omega,prime), prime) ]


def find_generator(q):
    order = q - 1
    order_prime_factors = prime_factors(order)
    
    for candidate in range(2, q):
        for factor in order_prime_factors:
            exponent = order // factor
            if pow(candidate, exponent, q) == 1:
                break
        else:
            return candidate

def prime_factors(n):
    factors = []
    divisor = 2
    while n > 1:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    return factors

def fast_shamir_share(secrets,num_coeffs,omega2,omega3, minimum=5,num_servers=5,prime = 800_011,order2=2,order3=3):
    small_values = [0] + secrets + [random.randrange(prime) for _ in range(num_coeffs)]

    small_coeffs = fft2_backward(small_values,omega2,prime)
    large_coeffs = small_coeffs + [0] * (order3-order2)
    large_values = fft3_forward(large_coeffs,omega3,prime)
    shares = large_values[1:]
    return shares

def decode_fast_shamir(shares,num_secrets,omega2,omega3,order2=2,order3=3,prime=800_011):
    large_values = [0] + shares
    large_coeffs = fft3_backward(large_values,omega3,prime)
    small_coeffs = large_coeffs[:order2]
    small_values = fft2_forward(small_coeffs,omega2,prime)
    secrets = small_values[1:num_secrets + 1]
    return secrets

def highest_power(n, k):
    if n == 0:
        return None  # Undefined for n=0
    
    power = 0
    while n % k == 0:
        n //= k
        power += 1

    return power

def rewrite(num):
    s = num - 1
    t = 0
    while s % 2 == 0:
        s = s // 2
        t += 1
    return s, t

def rabin_miller(num, iterations=10):
    s, t = rewrite(num)
    for _ in range(iterations):
        a = random.randrange(2, num - 1)
        v = pow(a, s, num)
        if v != 1:
            i = 0
            while v != (num - 1):
                if i == t - 1:
                    return False
                else:
                    i = i + 1
                    v = pow(v, 2, num)
    return True

def is_prime(num):
    if (num < 2): return False
    for prime in SMALL_PRIMES:
        if num == prime: return True
        if num % prime == 0: return False
    return rabin_miller(num)

def is_power_of(x, base):
    p = 1
    while p < x:
        p = p * base
    return p == x

def sample_prime(bitsize):
    lower = 1 << (bitsize-1)
    upper = 1 << (bitsize)
    while True:
        candidate = random.randrange(lower, upper)
        if is_prime(candidate):
            return candidate

def remove_factor(x, factor):
    while x % factor == 0:
        x //= factor
    return x

def prime_factor(x):
    factors = []
    for prime in SMALL_PRIMES:
        if prime > x: break
        if x % prime == 0:
            factors.append(prime)
            x = remove_factor(x, prime)
    assert(x == 1) # fail if we were trying to factor a too large number
    return factors

def find_prime(min_bitsize, order_divisor):
    while True:
        k1 = sample_prime(min_bitsize)
        for k2 in range(128):
            q = k1 * k2 * order_divisor + 1
            if is_prime(q):
                order_prime_factors  = [k1]
                order_prime_factors += prime_factor(k2)
                order_prime_factors += prime_factor(order_divisor)
                return q, order_prime_factors

def find_generator(q, order_prime_factors):
    order = q - 1
    for candidate in range(2, q):
        for factor in order_prime_factors:
            exponent = order // factor
            if pow(candidate, exponent, q) == 1:
                break
        else:
            return candidate
            
def find_prime_field(min_bitsize, order_divisor):
    q, order_prime_factors = find_prime(min_bitsize, order_divisor)
    g = find_generator(q, order_prime_factors)
    return q, g

def generate_parameters(min_bitsize, order2, order3):
    assert(is_power_of(order2, 2))
    assert(is_power_of(order3, 3))
    
    order_divisor = order2 * order3
    q, g = find_prime_field(min_bitsize, order_divisor)
    assert(is_prime(q))
    
    order = q - 1
    assert(order % order2 == 0)
    assert(order % order3 == 0)
    omega2 = pow(g, order // order2, q)
    omega3 = pow(g, order // order3, q)
    
    return q, omega2, omega3

def secret_share(secrets, num_servers):
    num_secrets = len(secrets)
    prime_bits = 80
    ORDER2 = num_servers + num_secrets + 1
    ORDER3 = 3*(num_servers + 1)
    Q, OMEGA2, OMEGA3 = generate_parameters(prime_bits, ORDER2, ORDER3)

    prime = Q
    order2 = ORDER2
    order3 = ORDER3
    omega2 = OMEGA2
    omega3 = OMEGA3
    num_coeffs = order2 - num_secrets - 1

    shares = fast_shamir_share(secrets,num_servers=num_servers,num_coeffs=num_coeffs,omega2=omega2,omega3=omega3,prime=prime,order2=order2,order3=order3)
    return (prime, omega2, omega3, order2, order3, shares)

def padded_params(num_server,num_secret):
    num_serv = power_of_3_greater_than_x(num_server) - 1
    y = power_of_2_greater_than_x(num_serv + num_secret) - 1

    while y > 3*num_serv:
        num_serv = 3*num_serv + 2
        y = power_of_2_greater_than_x(num_serv + num_secret) - 1

    num_secr = y - num_serv
    return num_serv,num_secr 

def split_list_into_k_parts(original_list, k):
    # Initialize k lists
    result = [[] for _ in range(k)]

    # Distribute elements into k parts
    for i, item in enumerate(original_list):
        result[i % k].append(item)

    return result

def reconstruct_list(num_servers):
    shares = [[] for _ in range(num_servers)]

    # Read each file and extract the shares
    for i in range(num_servers):
        with open(f"shares/share{i}.txt", "r") as file:
            line = file.readline().strip()  # Read the first line and remove leading/trailing whitespaces
            lst_str = line.split(":")[1]   # Extract the list part after the colon
            lst = eval(lst_str)            # Convert the string representation of list to a list
            shares[i] = lst                # Store the list in the corresponding share

    # Interleave the shares to reconstruct the original list
    reconstructed_list = []
    max_length = max(len(share) for share in shares)
    for i in range(max_length):
        for j in range(num_servers):
            if i < len(shares[j]):
                reconstructed_list.append(shares[j][i])

    return reconstructed_list

# def padded_params(num_server,num_secret):
#     limit = None
#     if(num_server > num_secret): 
#         limit = num_server
#     else:
#         limit = num_secret
#     power_3 = 1
#     power_2 = 1
#     first_time = True
#     while(1):
#         if first_time:
#             while(3**power_3 <= limit + 1):
#                 power_3 += 1
#             first_time = False
#         else: 
#             power_3 += 1
            
#         while 2**power_2 <= num_secret+3**power_3:
#             power_2 += 1
#         if(2*(3**power_3)-1 > 2**power_2):
#             break
#     num_serv = 3**power_3 - 1
#     num_secr = 2**power_2 -  3**power_3
    
#     return num_serv,num_secr 