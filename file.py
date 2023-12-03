# this file will contain all of the functions specified to implement and then attack learning with errors

# implementing learning with errors:
import numpy as np
import random
import galois
import traceback
import math 
import itertools
from itertools import combinations
import scipy

def key_gen(m,n,q):
    # A is a random matrix of size m x n
    A = np.random.randint((q), size=(m,n))
    # s is a random matrix of size n x 1
    s = np.random.randint((q), size=(n,1))

    # e is a matrix of size m x 1 that has random values from either (1,0,-1)
    e = np.random.choice([1,0,-1], (m,1))

    # Initialize the array with zeros
    e_3 = np.zeros((m, 1), dtype=int)

    # Randomly select two indices for 1 and -1 separately
    indices_ones = np.random.choice(m, size=2, replace=False)
    indices_neg_ones = np.random.choice(np.setdiff1d(np.arange(m), indices_ones), size=2, replace=False)

    # Assign 1 to the selected indices for 1 and -1
    e_3[indices_ones] = 1
    e_3[indices_neg_ones] = -1
    # calculate b = A.s + e
    b = (np.dot(A, s) + e) % q
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
    
    # now decrpt this function
    result = decrypt(ciphertext,s_int,q)
   # print("res",result)
    return result

def crack2(ciphertext, public_key, q):
    # define galois field
    gf = galois.GF(q)

    # intepret A and b from the public_key
    A = public_key[0]
    b = public_key[1]

    n = A.shape[1]
    m = A.shape[0]

    # while loop until we find the condition where count / num_errors = 4!

    num_errors = 0
    while num_errors != 4:
        # randomly select a public key pair; (A,b) and solve for s. Then check this value for every other 
        # value for all other values to find the error
        # Randomly select an index corresponding to a row, out of the m possible rows
        #random_row_index = np.random.choice(m)
        # gets m rows so we can work out a solution for s
        idx = np.random.choice(m, size=n, replace=False)

        # Get the randomly selected sqaure matrix and the corresponding b values
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
    

#def crack3(ciphertext, public_key, q):
    # see 19.3, breaking lwe:
    # use svp, to find e, as we have A,b -> solve by enumeration for crack 1 or 2?
    # look into gram schmdit 
    # recover shortest vector recovered using lattice reduction

    
    # use svp, to find e, as we have A,b -> solve by enumeration for crack 1 or 2?
    # look into gram schmdit 
    # recover shortest vector recovered using lattice reduction

    # define galois field
    gf = galois.GF(q)

    # intepret A and b from the public_key
    A = public_key[0]
    b = public_key[1]

    # convert A and b into the galois field, to allow for mod q operations
    A = gf(A)
    b = gf(b)
    #print(A)
  #  print(b.T)

    # creates a lattice basis
    # Create the bottom row [0, 0, ..., 1]
    # create n+1 zeros, and then n+1 = -1, under b
    bottom_row = np.zeros((1,A.shape[1] + 1))
   # print(bottom_row)
    # last value of row will be 1, which represents b
    bottom_row[0,-1] = 1
    #print(bottom_row)
    
    # Combine matrix A, vector b, and the bottom row
    # first n columns =  A, n +1 column = 1
    # bottom row is n 0s and then a 1 on the end
    # allows for the concatenation of A and b, and then has them vertically
    lattice_basis = np.vstack((np.column_stack((A, b)), bottom_row))
    print("lat", lattice_basis)
    lattice_basis = gf(lattice_basis)


    #lattice_matrix = np.block([[A, b], [np.zeros_like(A), np.eye(len(b))]])

    #print(lattice_matrix)


    # lattice basis has been correclty created;

    # to get the actual lattice, we then need to create all the lattice points from this basis
   # n = A.shape[1]
    #m = A.shape[0]

    # Enumerate through lattice points in the fundamental parallelepiped
    # This example uses a simple range for each dimension, you might need to adjust it based on your problem
    '''for coefficients in itertools.product(range(q), repeat=n):
        # Generate lattice point
        print(coefficients)
        coefficients = gf(coefficients)
        print(lattice_basis[:, :-1])
        lattice_point = np.dot(lattice_basis[:, :-1], coefficients)

        # Print or process the lattice point as needed
        print("Lattice Point:", lattice_point)'''
    n = A.shape[1]
    m = A.shape[0]
    #print("length vectors",n)

    # successfully created a lattice of size m+1, n+1
    #print(lattice_basis.shape)
    #print(A.shape[0]+1, A.shape[1]+1)

    # now iterate through all the columns of this basis to find the shortest vector?
    # dot product of each column and the vector s, +e should equal b?
    # 
 

    # want to get the shortest vector and its length,
    shortest_vector = None
    shortest_length = np.inf # set this to be massive and changed
    e_val = None
    # now from this basis we want to generate all possible values of integer combinations of basis vectors
    lattice_points = []
    # generates all possible integer combinations, of size n can have value of 0 to q
    # hence in galois finite field to allow for this mod
    for i in range(q**n):
        #print('here')
        # set up the point
        point = np.zeros(n+1, dtype=np.int32)
        for j in range(n):
            # allows us to determine the coefficients of the vector, based on i, j and q
            point[j] = (i % q**(j+1)) // q**j
        # turn it into the correcy field
        # add this so we can essentially add b whilst taking the dot product
        point[-1] = 1
        point = gf(point)
        #print("point", point)
        #lattice_points.append(point)
        print(point)

        # if here we check this against the lattice basis, this saves us time?
      #  print(m)
       # print('val',lattice_basis[:m,:n].shape)
        #print(x.shape)
        #print(point.shape)
        # to extract the value of A and perform the dot product to get value Ax+b
        minus_e = (np.dot(lattice_basis[:m],point))
        #As = np.dot(lattice_basis,point)
        print('As',minus_e)
        print('-As', gf(-minus_e))
        print(minus_e.shape)
        print('len',np.linalg.norm(minus_e))
        #b = gf(np.dot(b,-1))


        print(minus_e -(-minus_e))
        # to allow for the ring field, takes the inverse of -e = e and this will be the 
        # inverse mod q, so will return the distance to 0.
        # all values in result will be under q/2
        result= gf([min(x, y) for x, y in zip(minus_e, -minus_e)])
        print('re',result)
       # print('length', sum(result))
        print

        # bottom of s = -1, so have n values and then the minus -1 to esseentialy moinus b
       # and the value should be e
    #    print(b)
        #lattoce[:n]

        # make sure to tranpose b such that it is in the correct shape
        e = np.dot(A,point[:n]) - b.T
        print('e',e)
        print(minus_e == e)
        #e[e > q/2] = (q-e[0])
        print('new',e)
        
        '''#total = sum(row ** 2 for row in e)
        #norm = math.sqrt(total)
            # Flatten the nested list
        #flattened_vector = [item for sublist in e[0] for item in sublist]
        flattened_vector = e[0]
        print('flat',flattened_vector )
        # Calculate the sum of squared elements
        for x in flattened_vector:
            #if x > q/2:
                
            print('x',x-q)
        sum_of_squares = sum(x**2 for x in flattened_vector)
        
        # Calculate the square root of the sum of squares
        norm = sum_of_squares**0.5
        print(norm)'''
        # look for q-1s, 0, 1 and maybe 2, q-2?
        # calculate own norm -> as proper norm doesn't work
        # make sure norm is less than q/4


        # what happens when we take this and get a negative value?
     #   print('noe',e)
      #  print('e',-e)
        # works out the length of the vector, e which is the distance between As and b
        
        length =  np.linalg.norm(result)
        #print('len', length)
        leng_e = np.linalg.norm(-e)
        #print('noelength', leng_e)

        #if leng_e < length:
        #    length = leng_e
        if length < shortest_length:
            shortest_length = length
            shortest_vector = point
            e_val = result


    
    print('shortest', shortest_length, shortest_vector)
    print('e valeu',e_val)
    # checks all possible q^n pairs, which is what we want 
    print((q**n),'shape',len(lattice_points))
   # sval = gf(sval)
  #  e = np.dot(lattice_basis[:m,:n],sval) -b
   # len=np.linalg.norm(e)
    #print(len)
    #print(sval)
    res = decrypt(ciphertext, shortest_vector[:n], q)
    return res

        # problem is here
        #lattice_basis[:n] - b

        # distance between As, b is the norm of their difference which should also be e?
        # adds this to a list of points in the lattice, with varying coefficients 
        #lattice_points.append(point)
    
    #lattice = np.array(lattice_points)
    #print(lattice)

    #for s in lattice_points:
     #   result = np.dot(lattice_basis,s)
      #  print("res",result)
       # e = result - b.T
        #print(e)'''
    '''# Generate lattice points
     n = lattice_basis.shape[1] - 1  # Length of the lattice points
    print("n",n)
    lattice_points = []
    
    # use the extra value to store the value of y? but does this need to be 0 or
    # can this change?
    for i in range(q**(n)):
        # Set up the point
        point = np.zeros(n+1, dtype=np.int32)
        for j in range(n):
            # allows us to determine the coefficients of the vector, based on i, j and q
            point[j] = (i % q**(j+1)) // q**j
        point[-1] = 0  # Set the last component to 0 (for y)

        # Turn it into the correct field
        point = gf(point)
        #adds this to a list of points in the lattice, with varying coefficients 
        lattice_points.append(point)

    #lattice = np.array(lattice_points)
    # creates an array of all the possible vectors in 0, q-1
    lattice_points = np.array(lattice_points) # Transpose for correct shape

    print(lattice_points)
    print(lattice_points[-1])'''


    # Create the matrix here each column represents a lattice point transformed by the lattice basis
    #lattice_matrix = np.dot(np.block([[A, b], [np.zeros_like(A), np.eye(len(b))]]), lattice_points)


    #print('lattice',lattice_matrix)


    # Use lattice reduction algorithm to find short vectors
    # This step is not implemented in the provided code and requires a lattice reduction algorithm

    # TODO: Implement lattice reduction algorithm, such as LLL or BKZ

    # After lattice reduction, the last column of the reduced lattice should contain the solution 's'
    # Extract the solution 's' from the last column
    #s = lattice[-1]
    #print(s)

    # want to reduce this basis?
    #reduced, _ = np.linalg.qr(lattice)
    
    # A B
    #00 1 -> check lecture notes

    # Shortest vector problem: Given a basis B for L, find the nonzero vector v, with smallest norm in L

    # enumerate through the lattice to find shortest vector,e 
    '''shortest_vector = np.ones(lattice.shape[1]) *np.inf
    print("shortest", shortest_vector)
    # sets this to an arbitarily large variable such that we can use it for the comparison
    shortest_length = np.inf

    # loops through all vectors in the basis
    for i in range(lattice.shape[0]-1):
        print(i)
        # gets the vector
        vector = lattice[i]
        print("v", vector)
        # gets the length of each vector, but excludes the last element as this represents b
        length = np.linalg.norm(vector[:-1]) 
        print("len", length)
        print("vector", vector.dtype)
        vector =  vector.astype(np.int32)
        print(vector[-1])

        if length < shortest_length:
            shortest_length = length #- int(vector[-1])  #b[i]#vector[-1]
            shortest_vector = vector

    print(shortest_length)
    print(shortest_vector)
    
    s = shortest_vector[:-1]
    s_t = s.tranpose'''

    # Determine the search space for each dimension based on q
    '''search_space = [list(np.arange(q)) for _ in range(lattice.shape[1] - 1)]
    #search_space = list(search_space)
    print(search_space)
    # Iterate through every lattice point in the search space
    for point in np.ndindex(*map(len, search_space)):
        #print("p", point)
        vector = gf(point)
        #print('vector',vector)
        lattice_point = np.append(vector, 1)  # Include the last element (bottom row)
        #print("p",lattice_point)
        length = np.linalg.norm(lattice_point[:-1])  # Exclude the last element (bottom row)

        # Check if the lattice point is a solution
        print(np.dot(A, vector).T)
        print('b',b)
        print('reduced', reduced)
        print(np.array_equal(np.dot(A, vector), b))
        if np.array_equal(np.dot(A, vector), b):
            if length < shortest_length:
                shortest_length = length
                shortest_vector = lattice_point'''
        # Find a solution using the reduced lattice basis
    # Assuming that the last column of the reduced basis corresponds to the lattice vector [0, 0, ..., 1]
    ''' solution_coefficients = np.linalg.solve(reduced[:, :-1], -reduced[:, -1])

    # The last element of the solution is assumed to be 1, as it corresponds to the bottom row in the lattice basis
    solution_coefficients = np.append(solution_coefficients, 1)

    # Verify if the solution is valid
    if np.array_equal(np.dot(A, solution_coefficients[:-1]) % q, b):
        e = solution_coefficients

    #return None  # No solution found within the search space
    #e = shortest_vector
    print('e',e)
    
    s = [1,1,2,3]'''
    #s =  s.astype(np.int32)
    #print(s.dtype)

    #zprint()

    ''' # Extract dimensions
    n = A.shape[1]
    short = None
    # Try all possible candidate secret vectors
    for i in range(q**n):
        candidate_vector = np.array([i % q**(j+1) // q**j for j in range(n)], dtype=np.int32)
        candidate_vector = gf(candidate_vector)
        print(candidate_vector)
        # Check if A * candidate_vector is close to the ciphertext
        print("Aat",np.dot(A, candidate_vector))
        if np.array_equal((A @ candidate_vector), ciphertext):
            print(candidate_vector)
            short = candidate_vector

    print(short)'''


    #result = decrypt(ciphertext,short,q)
    #print(result)



    #return result

    '''import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Plot the vectors in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], marker='o', label='Vectors')
    ax.scatter(range(len(b)), b, 0, color='red', marker='x', label='Vector b')    
    ax.set_title('Vector Plot (3D)')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()
    plt.show() '''
    
    '''# define galois field
    gf = galois.GF(q)

    # intepret A and b from the public_key
    A = public_key[0]
    b = public_key[1]

    # convert A and b into the galois field, to allow for mod q operations
    A = gf(A)
    b = gf(b)
    print(A.shape)
    print(b)
    
    n = A.shape[1]
    lattice_points = []
    for i in range(q**(n)):
        # Set up the point
        point = np.zeros(n, dtype=np.int32)
        for j in range(n):
            # allows us to determine the coefficients of the vector, based on i, j and q
            point[j] = (i % q**(j+1)) // q**j
        #point[-1] = 0  # Set the last component to 0 (for y)

        # Turn it into the correct field
        point = gf(point)
        #adds this to a list of points in the lattice, with varying coefficients 
        lattice_points.append(point)

    #lattice = np.array(lattice_points)
    # creates an array of all the possible vectors in 0, q-1
    lattice_points = gf(lattice_points)
    print(lattice_points)

    # b = A.s + e
    #for x in range(0, A.shape[0]):
     #   print(A[x])
      #  print(b[x])
       # for s in lattice_points:
        #    print(s[:3])
         #   As = np.dot(A[x], s[:3].T)
          #  print("as", As)
           # e = As - b[x]
            #if e == 0:
                #break
            #print("e",e)

            # add all e values of 1, -1 and 0 to a list and then look through it this way
            # to work out the s-value'''
    '''shortest_vector = None
    # sets up length to be large, to allow for comparisons
    shortest_length = np.inf
    
    for s in lattice_points:
        print('A', A.shape)
        print('s',s.shape)
        As = np.dot(A,s.T)
       # As = np.array([gf.dot(row, s) for row in A])
        #print('as', As)
        print('b',b.T)

        As = np.array(As)
        print('as', As)
        #As_shaped = b.T.reshape(As.shape)
        print(As.shape)
        print(b.T.shape)
        b = np.array(b)
        print(As.shape == b.T.shape)
        print('no mod',As-b.T)
        # here, we need to take into account the negative values
        # anythinh bigger than 18, take away from 256 as it will be negative
        # not subtracting ptoperly? 
        e = (As -b.T) 
        print(e)
        r = np.where(e > 18, 256-e, e)
        print('e',r.shape)
        print('r',r)
        print('emod',e%q)

        norm_r = np.linalg.norm(r)
        norm_e = np.linalg.norm(e%q)
        print(norm_e == norm_r)
        print('norme', norm_e)
        print("norm",norm_r)
        if norm_r < shortest_length:
            shortest_length = norm_r
            shortest_vector = s



        #resulting_val = val_check(e)
        #print(resulting_val)

    print("shortest",shortest_vector, shortest_length)
    #s = np.linalg.solve(A,b)
     # Set up initial candidates (this is a simplification)

    result = decrypt(ciphertext, shortest_vector, q)
    return(result)'''
    


# tester function for crack2
def crack3(ciphertext, public_key, q):
    gf = galois.GF(q)

    # intepret A and b from the public_key
    A = public_key[0]
    b = public_key[1]

    n = A.shape[1]
    m = A.shape[0]
            # randomly select a public key pair; (A,b) and solve for s. Then check this value for every other 
        # value for all other values to find the error
        # Randomly select an index corresponding to a row, out of the m possible rows
        #random_row_index = np.random.choice(m)
        # gets m rows so we can work out a solution for s
    idx = np.random.choice(m, size=n, replace=False)

        # Get the randomly selected sqaure matrix and the corresponding b values
    random_matrix = gf(A[idx])
    random_b = gf(b[idx])
    print('rand', random_matrix)
    print('vector', random_b)

        # if the determinant of the square matrix is non-zero, matrix is full rank
        # as running into problems where the matrices weren't full rank and singular / invertible
    det = np.linalg.det(random_matrix)
    if det!= 0:   
            # solves the random system, to work out if sample s is the true s
        s = np.linalg.solve(random_matrix, random_b)

        print('s',s)


        num_errors =0
            # looks through each row A to find where the errors are
        for x in range(0, A.shape[0]):
                # works out A.s and if A.s = b then the error term must be 0
                # we know there are 4 error terms, so we stop when we reach this
            print('a',A[x], 'b',b[x])
            print(gf(A[x]) @gf(s))
            print(s.shape, A[x].shape, b[x].shape)
            if (gf(A[x]) @gf(s) != b[x]):
                    
                num_errors +=1
    print(num_errors,'num')
    
    # then once we have the correct s, we stop!
    res = decrypt(ciphertext, s, q)
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
    q = 32
    key, sval= key_gen(48,4,q)
    print(sval)
    ciphertext = encrypt(a, key,q)
    res5 = crack3(ciphertext, key,q)
    print('cipher',ciphertext)

    print('actual',a)
    print('result',res5)
    print(a == res5)
    print('sval',sval)
    #res6 = crack3_1(np.array([1,0,0,1,0,0,1,0,1,1,0,1,0,1]), key_gen(30,3,15),19)

    #res6 = key_gen(48,5,19)
