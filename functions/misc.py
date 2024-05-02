'''
Contains all the functions required for miscellaneous use-cases
'''

def is_float(string):
    '''
    Check if a string can be converted to float
    '''
    try:
        float(string)
        return True
    except ValueError:
        return False

def is_int(string):
    '''
    Check if a string can be converted to int
    '''
    try:
        int(string)
        return True
    except ValueError:
        return False