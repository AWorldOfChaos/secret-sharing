"""
Script to generate a convenient prime
"""

import random

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

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

ORDER2 = 512
ORDER3 = 729
Q, OMEGA2, OMEGA3 = generate_parameters(80, ORDER2, ORDER3)

print("Prime is %d" % Q)
print("%d-th principal root of unity is %d" % (ORDER2, OMEGA2))
print("%d-th principal root of unity is %d" % (ORDER3, OMEGA3))