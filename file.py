# this file will contain all of the functions specified to implement and then attack learning with errors

# implementing learning with errors:
import numpy as np
import random
#def key_gen(m,n,q):
    # A is a random matrix of size m x n

    # s is a random matrix of size n x 1

    # e is a matrix of size m x 1 that has random values from either (1,0,-1)

    # calculate b = Axs + e

    # A,B pub key

    # e is private key


def encrypt(plaintext, public_key, q):

    c = []
    # loops through each indivudal bit in the plaintext
    for bit in plaintext:
        print(bit)

        m =4
        # use a different r, generated here for each bit, random integer between 2 and m
        r = random.randint(2,m) # what is m tho?
        print(r)

        # access the values of A, b from the public key passing in respectively
        A = public_key[0]
        b = public_key[1]
        
        # generate random vector rT, that is a 1 x len(column A), matrix
        rT = np.random.randint(2, size = (1,len(A)))
        #print(rT)
        # performs the dot product using the random vector and the public key
        aT = np.dot(rT,A)
        #print(aT)

        # next perform the dot product using b
        b_prime = np.dot(rT, b)
        # then add pt * q/2
        #print("before",b_prime)
        pt = int(bit * (q/2))
        #print(pt)
        b_prime += pt
        #print("after",b_prime[0])
        b_prime = b_prime[0]

        # appends the array and b value to a list
        c.append([aT,b_prime])
    #print(c)
    # convers the list into an array, this is the ciphertext
    ciphertext = np.array(c)
   # print(ciphertext)
    return ciphertext

def decrypt(ciphertext, private_key, q):
    return 1

def crack1(ciphertext, public_key, q):
    return 2

def crack2(ciphertext, public_key, q):
    return 3

def crack3(ciphertext, public_key, q):
    return 4
    

#if __name__ == '__main__':
    #res = encrypt(np.array([0,1,1,0,0,1]),(np.array([2,3,4]),[1,1,2]),6)