# this file will contain all of the functions specified to implement and then attack learning with errors

# implementing learning with errors:
import numpy as np
import galois
import time
import math

def key_gen(m,n,q):
    # A is a random matrix of size m x n
    A = np.random.randint((q), size=(m,n))
    # s is a random matrix of size n x 1
    s = np.random.randint((q), size=(n,1))

    # e is a matrix of size m x 1 that has random values from either (1,0,-1)
    e = np.random.choice([1,0,-1], (m,1))

    # use this if we wanted to simulate the learning with few errors for crack2
    # Initialize the array with zeros
    e_3 = np.zeros((m, 1), dtype=int)

    # Randomly select two indices for 1 and -1 separately
    indices_ones = np.random.choice(m, size=2, replace=False)
    indices_neg_ones = np.random.choice(np.setdiff1d(np.arange(m), indices_ones), size=2, replace=False)

    # Assign 1 to the selected indices for 1 and -1
    e_3[indices_ones] = 1
    e_3[indices_neg_ones] = -1

    # use this if we wanted to simulate the learning with few errors for crack3, as
    # has half 0s, and then a quater of 1s, -1s
    # Initialize the array with zeros
    e_4 = np.zeros((m, 1), dtype=int)

    # Randomly select two indices for 1 and -1 separately
    indices_ones = np.random.choice(m, size=int(m/4), replace=False)
    indices_neg_ones = np.random.choice(np.setdiff1d(np.arange(m), indices_ones), size=int(m/4), replace=False)

    # Assign 1 to the selected indices for 1 and -1
    e_4[indices_ones] = 1
    e_4[indices_neg_ones] = -1
    # calculate b = A.s + e
    b = (np.dot(A, s) + e_4) % q
    # A,B pub key
    public_key = (A,b)

    # e is private key
    private_key = s
    print("s",s)

    return public_key, s

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
        #m=300 # find out how to not define this, as this will change?
        # use a different r, generated here for each bit, random integer between 2 and m
        #r = random.randint(2,m) # what is m tho?
        
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
    # defines the fields to allow for quicker computation in field q
    gf = galois.GF(q)
    # sets list for pt
    p = []
    # converts the private key into the galois field
    private_key = gf(private_key)
    for val in ciphertext:
        # converts values into galois field, so no need for mod
        a_prime_T = gf(val[0]%q) # remove this modq when we process it
        b_prime = gf(val[1])
        # v = a'T . s
        v = np.dot(a_prime_T, private_key)
        # m' = b' -v
        m_prime = b_prime - v

        # checks if the value is closer to q/2 or 0
        if (m_prime > q/4) and (m_prime <(3*q)/4):
            p.append(1)
        else:
            p.append(0)

    # converts to numpy array for the correct return format
    p_np = np.array(p)
    return p_np

def crack1(ciphertext, public_key, q):
    # as e is a matrix of all 0s, we are essentially solving the problem As = b.
    # splits public key down into its components, A and b
    A = (public_key[0])
    #print(A.properties)
    b = (public_key[1])


    # use galois to help create a fieldarray sublcass to work in the specific finite feild
    gf = galois.GF(q)

    # convert A and b into galois fields to allow for the correct solving of s
    A = gf(A)
    b = gf(b)

    # converts the matrices into the correct shapes to allow for solving of the system
    # use the value of n which is accessed via A.shape[1]
    A = A[:A.shape[1]]
    b = b[:A.shape[1]]

    # solve the linear system of equations As=b, giving us the value of s,the secret key
    # in the galois field
    s = np.linalg.solve(A,b)
    # Convert the solution back to integers, to allow for decryption
    s_int = s.astype(np.int32)
    
    # now decrypt this function
    result = decrypt(ciphertext,s_int,q)
    return result

def crack2(ciphertext, public_key, q):
    # define galois field
    gf = galois.GF(q)

    # intepret A and b from the public_key
    A = public_key[0]
    b = public_key[1]

    n = A.shape[1]
    m = A.shape[0]

    # while loop until we find the condition where num_errors = 4, as we know the system
    # has exactly 2 1s and 2 -1s, so this will produce 4 errors

    num_errors = 0
    while num_errors != 4:
        # gets n unique rows, to allow for us to have a square matrix A
        # can be in range up to m, as this is how many rows we have 
        idx = np.random.choice(m, size=n, replace=False)

        # Get the randomly selected sqaure matrix and the corresponding b values,
        # for each of the n values 
        random_matrix = gf(A[idx])
        random_b = gf(b[idx])

        # if the determinant of the square matrix is non-zero, matrix is full rank
        # as running into problems where the matrices weren't full rank and singular / invertible
        det = np.linalg.det(random_matrix)
        if det!= 0:   
            # solves the random system, to work out if sample s is the true s
            s = np.linalg.solve(random_matrix, random_b)

            num_errors =0
            # looks through each row A to find where the errors are
            for x in range(0, A.shape[0]):
                # works out A.s and if A.s = b then the error term must be 0
                # we know there are 4 error terms, so we stop when we reach this
                if (np.dot(gf(A[x]), s) != b[x]):
                    num_errors +=1


    # then once we have the correct s, we stop!
    res = decrypt(ciphertext, s, q)
    return res

def crack3(ciphertext, public_key, q):
    gf = galois.GF(q)

    # intepret A and b from the public_key
    A, b = public_key

    # puts the matrices into the correct field
    A = gf(A)
    b = gf(b)

    m,n = A.shape

    # create a lattice basis,
    # Create the bottom row [0, 0, ..., 1]
    # create n+1 zeros, and then n+1 = -1, under b
    bottom_row = np.zeros((1,A.shape[1] + 1))
    # last value of row will be 1, which represents b
    bottom_row[0,-1] = 1
    
    # Combine matrix A, vector b, and the bottom row
    # first n columns =  A, n +1 column = 1
    # bottom row is n 0s and then a 1 on the end
    # allows for the concatenation of A and b, and then has them vertically
    lattice_basis = np.vstack((np.column_stack((A, b)), bottom_row))
    lattice_basis = gf(lattice_basis)
    
    # define 'shortest variables'
    shortest_distance = np.inf
    shortest_vector = None
    
    # as this stays the same, saves doing the calculation?
    lattice_basis_m_n = lattice_basis[:m,:n]
    
    # look through x, do np.dot(Ax) +b = -e, where these values are this far from the lattice basis
    
    # add timeout loop, as it takes too long for medium input
    start_time = time.time()
    for i in range(q**n):
        # stops after 4 minutes and returns current best, for autograder!
        if time.time() > start_time +240:
            break
        # calculate the value x here, has a value between 0,q-1 and size n
        point = np.zeros(n, dtype=np.int32)
        for j in range(n):
            # allows us to determine the coefficients of the vector, based on i, j and q
            point[j] = (i % q**(j+1)) // q**j

        # put the point into the galois field,
        point = gf(point)


        # take the dot product of A and x = point, then subtract b.T, to allow for consistency
        e = np.array((np.dot(lattice_basis_m_n,point)) -b.T)

        # look through the error vector and work of the distance from 0 or q, if q//2, then work out for q - e, otherwirse 
        # distance = value, and then takes the sum of the squares of these distances
        # efficiently computes the sum, 
        sqrt = math.sqrt(np.sum(np.where(e > q / 2, (q - e)**2, e**2)))


        # then checks the distance against the current shortest, and if shorter will replace
        # with the current distance and the point is the new current shortest vector
        if sqrt < shortest_distance:
            shortest_distance = sqrt
            shortest_vector = point

    # then once we have the correct s, we stop!
    res = decrypt(ciphertext, shortest_vector, q)
    return res
                


if __name__ == '__main__':
    # n =16, m =300, q =53
    #res = encrypt(np.array([1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,1,0,1,0]), key_gen(300,16,53), 53)
    #print("resulting", (res))

    #array = np.array
    #res2 = decrypt(np.array([([23,3,2,3,3,3,3,3,3,33,3,3,4,5,5,6],23), ([23,3,32,3,3,3,3,3,3,33,3,3,4,5,5,6],33)],dtype='object'),key_gen(300,16,53), 53)
    #for i in res:
    #    print(i)

    #res4 =crack2(np.array([1,0,0,1,0,0,1,0,1,1,0,1,0,1]), key_gen(256,64,491), 491)
    a = [1,1,1,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,1,0]
    q = 19
    key, sval= key_gen(48,5,q)
    print(sval)
    ciphertext = encrypt(a, key,q)
    start = time.time()
    res5 = crack3(ciphertext, key,q)
    end = time.time()

    print('time;', end-start)
    #print('cipher',ciphertext)

    print('actual',a)
    print('result',res5)
    print(a == res5)
    print('sval',sval)
    #res6 = crack3_1(np.array([1,0,0,1,0,0,1,0,1,1,0,1,0,1]), key_gen(30,3,15),19)
    #res6 = key_gen(48,5,19)
