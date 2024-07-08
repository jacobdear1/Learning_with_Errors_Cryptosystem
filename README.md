## Learning with Errors Crytosystem

This project implemneted the Learning with Errors (LWE) cryptosystem, that aims to be robust to quantum computers.

Within the file, there are 6 functions. 

key_gen, generates the public and private keys, but only returns the public key

encrypt, encrypts the plaintext file based on the public_key and iteration of the plaintext

decrypt, decrypts the ciphertext based on the private_key and checks it has returned the correctly decoded value (correct private key)

crack1, decrypting the ciphertext when we are not provided with the private_key, but the matrix e is all 0.

crack2, decrypting the ciphertext when we have a few errors within the system, I found there to be 4, which helped my methodology

crack3, decrypting the ciphertext when we don't know any additional information, via a lattice basis approach and computation of the shortest error vector
