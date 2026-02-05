#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSA and digital signature

Introduction:
The public key encryption method is based on a private (secret), S, and public key, P,
and associated functions S() and P(). These functions should be easy to compute, however,
if one knows P(), it should be hard to computer S() from it.
The functions are inverses of one another, i.e, for a message M, we have
P(S(M)) = M
S(P(M)) = M

A standard public key encrypted communication scheme goes as follows. Bob encrypts a message, M,
using Alice's public function: C=P_A(M), known as the cypher. Bob sends the encrypted message Alice. Alice receives
the message and decrypts it using her private function S_A(C) = S_A(P_A(M)) = M.

RSA setup:
    The algorithm uses two different keys
    1. Public key - shared with everyone
    2. Private key - kept secret
    Data encrypted by the public key can only be decrypted with the private key
    and vice versa.

The algorithm includes the following steps:
    1. Pick two large primes: p and q
    2. Compute their product n = p * q
    3. Compute Euler's totient, phi(n), i.e, the number of number co-prime to n between 1 and n.
        phi(n) = (p-1)*(q-1)
    4. Chose a public exponent, 1 < e < phi(n), coprime to phi(n)
    5. Compute the private exponent d (modular inverse of e mod phi(n))

    Public key: (e,n)
    Private key: (d,n)

Encryption: sending a message m<n, the sender evaluates c = m^e mod n.
Decryption: receiver computes m = c^d mod n.

The algorithm works since e*d = 1 mod phi(n), therefore
e*d = 1 phi(n)*k, as a result m^{ed} = m*(m^{phi(n)})^k, Euler's theorem states that
m^phi(n) = 1 mod n, therefore m^{ed} = m mod n

Digital signatures:
Alice computes a digital signature sigma=S_A(M), and sends it to Bob. Bob evaluates
P_A(sigma) = P_A(S_A(M)) = M.


Complexity:
    Assuming, e = O(1), and, lg(d), lg(n) < beta.
    The encryption and decryption uses O(1) and O(beta) modular multiplications,
    correspondingly. Each modular multiplications take O(beta^2) bit operations
    (using the standard (non-recursive) bit-multiplication method).
"""
from modular_exponentiation import modular_exponentiation_iterative as modExp

class RSA(object):
    """Simple implementation of the RSA algorithm"""

    def __init__(self, n, e, d):
        self.public_key = (e, n)
        self.private_key = (d, n)

    def encrypt(self, message: int) -> int:
        """
        Encodes the message using the public key
        Args:
            message (int): message to be encrypted

        Returns:
            int: the encrypted message
        """
        e, n = self.public_key
        return modExp(message, e, n)

    def decrypt(self, encrypted_message: int) -> int:
        """
        Decodes the message using the private key
        Args:
            encrypted_message (int), the message to be decrypted

        Returns:
            int: decrypted message
        """
        d, n = self.private_key
        return modExp(encrypted_message, d, n)

    def digital_signature(self, message: int) -> int:
        """
        Computes the digital signature using the private key
        """
        return self.decrypt(message)

    def decrypt_signature(self, digital_signature: int) -> int:
        """
        Computes the digital signature using the public key
        """
        return self.encrypt(digital_signature)





if __name__ == "__main__":
    message = 10
    p = 11
    q = 17
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 7
    d = 23
    rsa = RSA(n, e, d)
    encrypted_message = rsa.encrypt(message)
    decrypted_message = rsa.decrypt(encrypted_message)

    digital_signature = rsa.digital_signature(message)
    decrypt_signature = rsa.decrypt_signature(decrypted_message)

    print('------------ TESTS ------------')
    print(f'RSA test: {message==decrypted_message}')
    print(f'Digital signature test: {digital_signature==decrypt_signature}')


