# Learning with Errors Cryptosystem

This project implemented the Learning with Errors (LWE) cryptosystem, which aims to be robust against quantum computers.

Within the file, there are six functions:

- `key_gen()`: Generates the public and private keys, but returns only the public key.
- `encrypt()`: Encrypts the plaintext file using the public_key and iteration of the plaintext.
- `decrypt()`: Decrypts the ciphertext using the private_key and checks that it has returned the correctly decoded value (with the correct private key).
- `crack1()`: Decrypts the ciphertext without the private_key, assuming the matrix `e` is all zeros.
- `crack2()`: Decrypts the ciphertext with a few errors in the system. I found there to be four, which helpe my methodology.
- `crack3()`: Decrypts the ciphertext without knowing any additional information, via a lattice basis approach and the computation of the shortest error vector.
