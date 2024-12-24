import numpy as np
from midterm2.functions import gauss_iter_solve

x_data = np.array([
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2020])

y_data = np.array([
    390.1,
    391.85,
    394.06,
    396.74,
    398.81,
    401.01,
    404.41,
    406.76,
    408.72,
    411.65,
    414.21])




def main():
    #----------------------------------------------------------------------------------------------------------------
    # Question 1 (c)

    spline_coeff = np.array([
        [-1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 4, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 4, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 4, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 4, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1], 
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1]
    ])

    # RHS vector taken from the excel sheet

    RHS_vector = np.array([
        0, 
        1.38, 
        1.41, 
        -1.83, 
        0.39, 
        3.6, 
        -3.15, 
        -1.17, 
        2.91, 
        -1.11, 
        0
    ])

    # Solve the system using Gauss-Seidel algorithm
    c_coefficients = gauss_iter_solve(spline_coeff, RHS_vector, alg='seidel')
    print(f'The calculated solution is: {c_coefficients}')

    # Check the result using the numpy function
    print(f'The confirmed solution using a numpy function is: {np.linalg.solve(spline_coeff, RHS_vector)}')

    
    # Question 1 (d)
    # Find the d coefficients
    d_i = np.diff(c_coefficients) / 3
    print(f'The d coefficients of the cubic spline are: {d_i}')

    # Find the b coefficients
    div_diff = np.diff(y_data) / np.diff(x_data)
    
   
    b_i = div_diff - c_coefficients[:-1] - d_i
    print(f'The b coefficients of the the cubic spline are: {b_i}')

    # Find the a coefficients
    a_i = y_data[:-1]           # Losing the last value 
    print(f'The a coefficients of the cubic spline are: {a_i}')               

    # Now find the interpolated value of March 2015 (ie: x = 2015.25)
    # use the target point x = 2015.25
    # use the basepoint x_i = 2015 where i = 5
    s3_5 = a_i[5] + b_i[5] * (2015.25 - x_data[5]) + c_coefficients[5] * (2015.25 - x_data[5]) ** 2 + d_i[5] * (2015.25 - x_data[5]) ** 3
    print(f'The interpolated value of CO2 concentration from March 2015 is {s3_5}')
    #-------------------------------------------------------------------------------------------------------------------------------


    return





if __name__ == "__main__":
    main()