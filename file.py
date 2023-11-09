# this file will contain all of the functions specified to implement and then attack learning with errors

# implementing learning with errors:
import numpy as np
import random
def key_gen(m,n,q):
    # A is a random matrix of size m x n
    A = np.random.randint((q), size=(m,n))
    # s is a random matrix of size n x 1
    s = np.random.randint((q), size=(n,1))

    # e is a matrix of size m x 1 that has random values from either (1,0,-1)
    e = np.random.choice([1,0,-1], (m,1))
    # calculate b = A.s + e
    b = (np.dot(A, s) + e) % q
    print(b)

    # A,B pub key
    public_key = (A,b)

    # e is private key
    private_key = s

    return private_key

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
    # check that the ciphertext and the private key are compatible sizes?   
    print(private_key)

    # sets list for pt
    p = []
    #for val in ciphertext:
    for val in range(0, len(ciphertext)):
        print("val", ciphertext[val][1])
        # turn a' back into a numpy array to allow for the dot product
        a_prime_T = np.array(ciphertext[val][0])
        print("here",a_prime_T)
        # v = a'T . s, then adding the mod q to it
        v = np.dot(a_prime_T, private_key) % q
        print("makes it")
        print(v % q)
        # m' = b' -v
        b_prime = np.array(ciphertext[val][1])
        print("b",b_prime)
        print(type(v))
        print(type(b_prime))
        print(b_prime-v)
        m_prime = b_prime - v
        print("m",m_prime[0])

        m = abs(0-m_prime)
        q_over_2 = abs(0-((q/2)%q))
        print(m,q_over_2)
        print(m-q_over_2)

        if m - q_over_2 <=0:
            p.append(0)
        else:
            p.append(1)
            
    print("p",p)
    return p

def crack1(ciphertext, public_key, q):
    return 2

def crack2(ciphertext, public_key, q):
    return 3

def crack3(ciphertext, public_key, q):
    return 4
    

if __name__ == '__main__':
    # n =16, m =300, q =53
    #res = encrypt(np.array([1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,1,0,1,0]), key_gen(300,16,53), 53)
    #print("resulting", (res))

    #array = np.array
    res2 = decrypt(np.array([([23,3,2,3,3,3,3,3,3,33,3,3,4,5,5,6],23), ([23,3,32,3,3,3,3,3,3,33,3,3,4,5,5,6],33)],dtype='object'),key_gen(300,16,53), 53)
    #for i in res:
    #    print(i)