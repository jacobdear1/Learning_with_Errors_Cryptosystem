# this file will contain all of the functions specified to implement and then attack learning with errors

# implementing learning with errors:
import numpy as np
import random
from scipy.linalg import solve 
from sympy import solve_linear_system
def key_gen(m,n,q):
    # A is a random matrix of size m x n
    A = np.random.randint((q), size=(m,n))
    # s is a random matrix of size n x 1
    s = np.random.randint((q), size=(n,1))

    # e is a matrix of size m x 1 that has random values from either (1,0,-1)
    e = np.random.choice([1,0,-1], (m,1))
    # calculate b = A.s + e
    b = (np.dot(A, s) + e) % q

    # A,B pub key
    public_key = (A,b)

    # e is private key
    private_key = s

    return public_key

def encrypt(plaintext, public_key, q):

    # Define the data type for the structured array
    dtype = np.dtype([('array', np.ndarray), ('b', int)])

    # Create a NumPy object-type array with pairs of arrays and an integer 'b'
    array_length = len(plaintext)  # Define the length of the array, same as length of pt
    ciphertext = np.empty(array_length, dtype=dtype) 

    # empty arrays to allow for the values to be built as we find them
    s_arrays = []
    b_values = []

    # access the values of A, b from the public key passing in respectively
    A = public_key[0]
    b = public_key[1]

    # m = A.shape[0]

    # loops through each indivudal bit in the plaintext
    for bit in plaintext:
        # not sure these are necessary!
        m=300 # find out how to not define this, as this will change?
        # use a different r, generated here for each bit, random integer between 2 and m
        r = random.randint(2,m) # what is m tho?
        
        # generate random vector rT, that is a 1 x len(column A), matrix
        rT = np.random.randint(2, size = (1,len(A)))

        # performs the dot product using the random vector and the public key
        aT = np.dot(rT,A)
        # so the array is correctly formatted
        aT_1 = aT[0]
        # appends to list, that allows us to read into numpy array
        s_arrays.append(aT_1)

        # next perform the dot product using b
        b_prime = np.dot(rT, b)
        # then add pt * q/2
        pt = int(bit * (q/2)) # need to check this, what happens if q is .5, do we floor or ceiling?
        # add pt to b_prime
        b_prime += pt
        # performs mod q to be, as specified in lecture notes
        b_prime %= q
        # flattens the numpy array
        b_int_f = b_prime.flatten()
        # then gets rid of the extra braces to allow for single element
        b_int = int(''.join(map(str, b_int_f)))
        # appends each value to a list to allow for us to read into numpy array
        b_values.append(b_int)

    # assigns the generated values to the structured array
    ciphertext['array'] = s_arrays
    ciphertext ['b'] = b_values

    # this will have the same number of values as the ciphertext, as one corresponds to this
    return ciphertext

def decrypt(ciphertext, private_key, q):
    # sets list for pt
    p = []
    for val in ciphertext:
        # turn a' back into a numpy array to allow for the dot product
        a_prime_T = np.array(val[0])
        # v = a'T . s, then adding the mod q to it
        v = np.dot(a_prime_T, private_key) % q
        # m' = b' -v
        # turns the integer into a numpy array to allow for operations
        b_prime = np.array(val[1])
        m_prime = (b_prime - v) % q

        if (m_prime > q/4) and (m_prime <(3*q)/4):
            p.append(1)
        else:
            p.append(0)

    p_np = np.array(p)
    return p_np

def crack1(ciphertext, public_key, q):
    # see 19.3, breaking lwe:
    # use svp, to find e, as we have A,b -> solve by enumeration for crack 1 or 2?
    # look into gram schmdit 
    # recover shortest vector recovered using lattice reduction
    
    # as e is a matrix of all 0s, we are essentially solving the problem As = b.
    A = public_key[0]
    b = public_key[1]

    # converts the matrices into the correct shapes to allow for solving of the system
    # use the value of n which is accessed via A.shape[1]
    A = A[:A.shape[1]]
    b = b[:A.shape[1]]

    # change these to the correct fields -> using galios function to make sure integers are returned
    print(A.shape)
    s = np.linalg.solve(A,b)
    print(len(s))
    print(s)
    print(np.allclose(np.dot(A,s),b))
    print(ciphertext)

    # now decrpt this function
    result = decrypt(ciphertext,s,q)
    print(result)


    return result

def crack2(ciphertext, public_key, q):
    return 3

def crack3(ciphertext, public_key, q):
    return 4
    
#if __name__ == '__main__':
    # n =16, m =300, q =53
    #res = encrypt(np.array([1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,1,0,1,0]), key_gen(300,16,53), 53)
    #print("resulting", (res))

    #array = np.array
    #res2 = decrypt(np.array([([23,3,2,3,3,3,3,3,3,33,3,3,4,5,5,6],23), ([23,3,32,3,3,3,3,3,3,33,3,3,4,5,5,6],33)],dtype='object'),key_gen(300,16,53), 53)
    #for i in res:
    #    print(i)

    #res3 =crack1(np.array([1,0,0,1,0,0,1,0,1,1,0,1,0,1]), key_gen(256,64,491), 491)