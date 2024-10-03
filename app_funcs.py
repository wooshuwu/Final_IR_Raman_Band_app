#@title *Now run symmetry operations as arrays*

import sympy as sp
import math
from numpy.linalg import multi_dot
import numpy as np
import pandas as pd
import re

###These are the some symmetry operations

def file_matrix(file):
    split=file.split()
    mat=np.array(split[2:])
    row=int(len(mat)/4)
    matrix=mat.reshape(row,4)
    return(matrix)

def open_file(file, fold_name):
  path= f"{fold_name}/{file}.xyz"
  # fold_name +file+".xyz"
  with open(path) as ifile:
    file_name = "".join([x for x in ifile])
  return file_name

def format_molecules(input_string):
    # Define a regular expression pattern to match numbers
    pattern = r'\d+'
    
    # Use the re.sub() function to replace numbers with _{number}_
    new_string = re.sub(pattern, lambda match: f"_{{{match.group()}}}", input_string)
    
    return new_string

def format_vsepr_latex(s):
    result = ""
    for char in s:
        if char.isdigit():
            result += "_{{{}}}".format(char)
        else:
            result += char
    return result
  
def format_rotational_latex(df, column_name):
  """format rotational column with dataframe"""
  # Create a new list to store the formatted items
  formatted_items = []
  
  # Iterate over each item in the column
  for item in df[column_name]:
    if(pd.notnull(item)):
      components = str(item).split(",")
      # print(f"components: {components}")
      str_all = ""
      for component in components:
        component = component.replace(" ", "")
        str_all += f"{component[0]}_{{{component[1:]}}}"
      formatted_items.append(f"${str_all}$")
    else:
      formatted_items.append(item)
  column = df[column_name]
  return pd.Series(formatted_items, index=column.index)

def format_char_table_header(s: str) -> str:
    # Replace "inf" with "\infty"
    if s not in ("Linear", "Rotational", "Quadratic"):
        # Apply subscript formatting first
        if len(s) > 1:
            symbol = s[0]
            subscript = s[1:]
            
            # Replace patterns in the symbol part
            replace_patterns = {
                "inf": r"\infty",
                "s": r"\sigma",
                "S": r"\sigma"
            }
            for pattern_og, pattern_new in replace_patterns.items():
                symbol = symbol.replace(pattern_og, pattern_new)
            
            # Format the result with subscript
            return f"{symbol}_{{{subscript}}}"
        else:
            # For single character strings, just apply replacements
            replace_patterns = {
                "inf": r"\infty",
                "s": r"\sigma",
                "S": r"\sigma"
            }
            for pattern_og, pattern_new in replace_patterns.items():
                s = s.replace(pattern_og, pattern_new)
            return s
    else:
        return s


def format_superscripts(s):
    result = ""
    for char in s:
        if char.isdigit():
            result += "^{{{}}}".format(char)
        else:
            result += char
    return result

def format_subscript_latex(s: str) -> str:
    # Replace "inf" with "\infty"
    s = s.replace("inf", r"\infty")
    
    # Apply subscript formatting
    if len(s) > 1:
        return f"{s[0]}_{{{s[1:]}}}"
    else:
        return s
    
def format_subscript_latex_title(s: str) -> str:
    # Replace "inf" with "\infty"
    s = s.replace("inf", r"\infty")
    
    if(s not in ("Linear", "Rotational", "Quadratic")):
    
        # Apply subscript formatting
        if len(s) > 1:
            return f"{s[0]}_{{{s[1:]}}}"
        else:
            return s    
    else:
        return s
    
def format_superscripts_from_df_column(df, column_name):
    """
    Function to process non-NA values in a DataFrame column.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        column_name (str): The name of the column to process.
        
    Returns:
        pandas.Series: A Series containing the processed non-NA values.
    """
    # Get the column as a Series
    column = df[column_name]
    processed_values = []
    
    superscript_pattern = r"(\d+)"
    # latex_superscript = "^{{{}}}"
    
    for value in column:
        if pd.notnull(value):
            processed_value = re.sub(superscript_pattern, r"^{\1}", value)
            processed_values.append(f"${processed_value}$")
        else:
            processed_values.append(value)
    
    return pd.Series(processed_values, index=column.index)
    
def identity_matrix(atoms):
  E=np.identity(3)
  E_matrix=np.dot(atoms,E)
  return(E,E_matrix)


def c4_matrix(atoms): #This is a 90 degree proper rotation on Z-axis
  C4=np.array([[round(sp.cos(sp.pi/2),4),(round(-sp.sin(sp.pi/2), 4)),0],[round(sp.sin(sp.pi/2), 4),(round(sp.cos(sp.pi/2), 4)),0],[0,0,1]])
  c4_matrix=np.dot(atoms,C4)
  return(C4,c4_matrix)

def c4y_matrix(atoms): #This is a 90 degree proper rotation on y-axis
  C4y=np.array([[round(sp.cos(sp.pi/2),4),0,(round(-sp.sin(sp.pi/2), 4))],[0,1,0],[round(sp.sin(sp.pi/2), 4),0,(round(sp.cos(sp.pi/2), 4))]])
  c4y_matrix=np.dot(atoms,C4y)
  return(C4y,c4y_matrix)

def c2_matrix_z(atoms): #This is a 180 degree proper rotation on Z-axis
  C2=np.array([[sp.cos(sp.pi),-sp.sin(sp.pi),0],[sp.sin(sp.pi),sp.cos(sp.pi),0],[0,0,1]])
  c2_matrix=np.dot(atoms,C2)
  return(C2,c2_matrix)

def c2_x(atoms): #This is a 180 degree proper rotation on x-axis
  C21=np.array([[1,0,0],[0,sp.cos(sp.pi),-sp.sin(sp.pi)],[0,sp.sin(sp.pi),sp.cos(sp.pi)]])
  c2_prime_matrix=np.dot(atoms,C21)
  return(C21,c2_prime_matrix)

def c2_y(atoms): #This is a 180 degree proper rotation on y-axis
  C21y=np.array([[sp.cos(sp.pi),0,-sp.sin(sp.pi)],[0,1,0],[sp.sin(sp.pi),0,sp.cos(sp.pi)]])
  c2y_prime_matrix=np.dot(atoms,C21y)
  return(C21y,c2y_prime_matrix)

def c2_dprime(atoms): #This is a 180 degree proper rotation in between the axis
  c2_dprime=np.array([[0,-1,0],[-1,0,0],[0,0,-1]])
  c2_dprime_matrix=np.dot(atoms,c2_dprime)
  return(c2_dprime,c2_dprime_matrix)
  #c2 on z-axis

def c3_matrix_z(atoms): # This is a C3 (120 degree) rotation on the Z-axis
  C3=np.array([[round(sp.cos(sp.pi/1.5), 8),(round(-sp.sin(sp.pi/1.5), 8)),0],[round(sp.sin(sp.pi/1.5),8),(round(sp.cos(sp.pi/1.5), 8)) ,0],[0,0,1]])
  c3_matrix=np.dot(atoms,C3)
  return(C3,c3_matrix)
"""def c3_matrix_z(): # The result is thesame as the above commented lines of code
  theta = 2 * math.pi / 3
  c3 = np.array([[-math.cos(theta), -math.sin(theta), 0],
                           [math.sin(theta), -math.cos(theta), 0],
                           [0, 0, 1]])
  c3_matrix=np.dot(atoms,c3)
  return(c3,c3_matrix)"""


def c_3_matrix_x(atoms): # This is a C3 (120 degree) rotation on x-axis
  C_3=np.array([[1,0,0],[0,round(sp.cos(sp.pi/1.5), 4),(round(-sp.sin(sp.pi/1.5), 4))],[0,round(sp.sin(sp.pi/1.5),4),(round(sp.cos(sp.pi/1.5),4))]])
  c_3_matrix=np.dot(atoms,C_3)
  return(C_3,c_3_matrix)

def c_3_matrix_y(atoms): # This is a C3 (120 degree) rotation on y-axis
  C_32=np.array([[round(sp.cos(sp.pi/1.5), 4),0,(round(-sp.sin(sp.pi/1.5), 4))],[0,1,0],[round(sp.sin(sp.pi/1.5),4),0,(round(sp.cos(sp.pi/1.5),4))]])
  c_32_matrix=np.dot(atoms,C_32)
  return(C_32,c_32_matrix)

def c3_dprime(atoms): # This is a C3 (120 degree) rotation between the axis
  C32=np.array([[0,1,0],[0,0,1],[1,0,0]])
  c3_dprime=np.dot(atoms,C32)
  return(C32,c3_dprime)

def inversion(atoms):
   E=np.identity(3)
   I_matrix=np.dot(atoms,-E)
   return(-E,I_matrix)

def s2(atoms): # This is a 180 degree improper rotation along the Z-axis
  C2=np.array([[sp.cos(sp.pi),-sp.sin(sp.pi),0],[sp.sin(sp.pi),sp.cos(sp.pi),0],[0,0,1]])
  sh=np.array([[1,0,0],[0,1,0],[0,0,-1]])
  s2=np.dot(C2,sh)
  s2_matrix=multi_dot([atoms,C2,sh])
  return(s2,s2_matrix)

def s3(atoms): # This is a 120 degree improper rotation along the Z-axis
  C3=np.array([[round(sp.cos(sp.pi/1.5), 4),(round(-sp.sin(sp.pi/1.5), 4)),0],[round(sp.sin(sp.pi/1.5),4) ,(round(sp.cos(sp.pi/1.5), 4)) ,0],[0,0,1]])
  sh=np.array([[1,0,0],[0,1,0],[0,0,-1]])
  s3=np.dot(C3,sh)
  s3_matrix=multi_dot([atoms,C3,sh])
  return(s3,s3_matrix)

def s_4(atoms):  #This is a 90 degree improper rotation in between the axis
  C_4=np.array([[round(sp.cos(sp.pi/2),4),(round(-sp.sin(sp.pi/2), 4)),0],[round(-sp.sin(sp.pi/2), 4),(round(sp.cos(sp.pi/2), 4)),0],[0,0,-1]])
  sh=np.array([[1,0,0],[0,-1,0],[0,0,1]])
  s_4=np.dot(C_4,sh)
  s_4_matrix=multi_dot([atoms,C_4,sh])
  return(s_4,s_4_matrix)

def c_4_matrix(atoms):  # 90 degree proper rotation in between the axis
  C_4=np.array([[round(sp.cos(sp.pi/2),4),(round(-sp.sin(sp.pi/2), 4)),0],[round(-sp.sin(sp.pi/2), 4),(round(sp.cos(sp.pi/2), 4)),0],[0,0,-1]])
  c_4_matrix=np.dot(atoms,C_4)
  return(C_4,c_4_matrix)

def s4(atoms): # This is the 90 degree improper rotation on the Z-axis.
  C4=np.array([[round(sp.cos(sp.pi/2),4),(round(-sp.sin(sp.pi/2), 4)),0],[round(sp.sin(sp.pi/2), 4),(round(sp.cos(sp.pi/2), 4)),0],[0,0,1]])
  sh=np.array([[1,0,0],[0,1,0],[0,0,-1]])
  s4=np.dot(C4,sh)
  s4_matrix=multi_dot([atoms,C4,sh])
  return(s4,s4_matrix)

def s6(atoms): # This is the 60 degree improper rotation on the Z-axis.
  C6=np.array([[round(sp.cos(sp.pi/3),4),(round(-sp.sin(sp.pi/3), 4)),0],[round(sp.sin(sp.pi/3), 4),(round(sp.cos(sp.pi/3), 4)),0],[0,0,1]])
  sh=np.array([[1,0,0],[0,1,0],[0,0,-1]])
  s6=np.dot(C6,sh)
  s6_matrix=multi_dot([atoms,C6,sh])
  return(s6,s6_matrix)

def s62(atoms): # This is the 60 degree improper rotation in between the axis.
  s62=np.array([[0,1,0],[0,0,1],[-1,0,0]])
  s62_matrix=np.dot(atoms,s62)
  return(s62,s62_matrix)

def sigmah(atoms): #This is the horizontal reflection
  sh=np.array([[1,0,0],[0,1,0],[0,0,-1]])
  sigmah=np.dot(atoms,sh)
  return(sh,sigmah)

def sigmav(atoms): # This is the vertical reflection plane along the y-axis and also contains the principal axis
  sv=np.array([[-1,0,0],[0,1,0],[0,0,1]])
  sigmav=np.dot(atoms,sv)
  return(sv,sigmav)

def sigmav_xz(atoms): # This is the vertical reflection plane along the X- axis and also contains the principal axis
  Sv=np.array([[1,0,0],[0,-1,0],[0,0,1]])
  sigmav_xz=np.dot(atoms,Sv)
  return(Sv,sigmav_xz)

def sigmav_xy(atoms):
  Sv=np.array([[1,0,0],[0,1,0],[0,0,-1]])
  sigmav_xy=np.dot(atoms,Sv)
  return(Sv,sigmav_xy)

def sigmad(atoms): # This is the diagonal/dihedral reflection plane (A vertical mirror plane that bisects the angle between two C2 axes)
  sd=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
  sigmad=np.dot(atoms,sd)
  return(sd,sigmad)

def sigmad1(atoms): # This is the diagonal/dihedral reflection plane between the axis (reflection along the diagonal of a 3D space, which goes through the origin which goes through the origin and two axes)
  sd1=np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
  sigmad1=np.dot(atoms,sd1)
  return(sd1,sigmad1)

#Determines how many unmoved atoms are left in each symmetry operation
def unmoved_atoms_count(symmetry, atoms):
  """
  Determines how many unmoved atoms are left in each symmetry operation
  
  Args:
      symmetry (_type_): _description_
      atoms (_type_): _description_

  Returns:
      _type_: _description_
  """
  rho=0
  row_index=0
  for current_row in symmetry:
    match_bool = np.array_equal(current_row, atoms[row_index])
    # print(f"EQUAL: {match_bool} Rho: {rho} current row: {current_row}, atoms[row_index]: {atoms[row_index]}")
    if np.array_equal(current_row, atoms[row_index]):
        rho += 1
        # print(f"i: {current_row} rho: {rho}")
    
    row_index += 1
  #   print()
  # print(f"final rho: {rho}")  
  return rho

def apply_symm(pg, natoms, atoms):
    """
    Apply the symmetry operations on your chosen molecule and find out the number of unmoved positions*
    #code to assign symmetry operations to each point group and then return the resulting matrix.
    #def symmetry_designation:

    Args:
        pg (str): point group
    """
    if pg == "Dinf_h":
        E,E_1x=identity_matrix(atoms)
        C2,C2_1=c2_matrix_z(atoms)
        C21,C21_1=c2_x(atoms)
        c22,c22_1=c2_dprime(atoms)
        I,I_1= inversion(atoms)
        Sv,Sv_1=sigmav(atoms)
        Sv1,Sv1_1=sigmav_xz(atoms)
        Sv2,Sv2_1=sigmav_xy(atoms)
        group_mat = np.array([unmoved_atoms_count(E_1x, atoms),unmoved_atoms_count(C2_1, atoms),unmoved_atoms_count(C21_1, atoms),unmoved_atoms_count(c22_1, atoms),unmoved_atoms_count(I_1, atoms),unmoved_atoms_count(Sv2_1, atoms),unmoved_atoms_count(Sv1_1, atoms),unmoved_atoms_count(Sv_1, atoms)])

    elif pg == "D3h":
        E,E_1=identity_matrix(atoms)
        C3,C3_1=c3_matrix_z(atoms)
        C21,C21_1=c2_y(atoms)
        sh,sh_1=sigmah(atoms)
        S3,S3_1=s3(atoms)
        Sv,Sv_1=sigmav(atoms)
        group_mat = np.array([unmoved_atoms_count(E_1, atoms),unmoved_atoms_count(C3_1, atoms),unmoved_atoms_count(C21_1, atoms),unmoved_atoms_count(sh_1, atoms),unmoved_atoms_count(S3_1, atoms),unmoved_atoms_count(Sv_1, atoms)])

    elif pg =="Oh":
        E,E_1=identity_matrix(atoms)
        C4,C4_1=c4_matrix(atoms)
        C32,C32_1=c3_dprime(atoms)
        C2,C2_1=c2_matrix_z(atoms)
        C21,C21_1=c2_x(atoms)
        c22,c22_1=c2_dprime(atoms)
        I,I_1= inversion(atoms)
        S4,S4_1=s4(atoms)
        S62,S62_1=s62(atoms)
        sh,sh_1=sigmah(atoms)
        sv,sv_1=sigmav(atoms)
        sd,sd_1=sigmad(atoms)
        print(f"I: \n{I_1} \natoms: \n{atoms} \n{unmoved_atoms_count(I_1, atoms)}")
        group_mat = np.array([unmoved_atoms_count(E_1, atoms),unmoved_atoms_count(C32_1, atoms),unmoved_atoms_count(c22_1, atoms),unmoved_atoms_count(C4_1, atoms),unmoved_atoms_count(C21_1, atoms),unmoved_atoms_count(I_1, atoms),unmoved_atoms_count(S4_1, atoms),unmoved_atoms_count(S62_1, atoms),unmoved_atoms_count(sh_1, atoms),unmoved_atoms_count(sd_1, atoms)])

    elif pg == "C2v":
        E,E_1=identity_matrix(atoms)
        C2,C2_1=c2_matrix_z(atoms)
        sv,sv_1=sigmav(atoms)
        Sv,Sv_1=sigmav_xz(atoms)
        # print(f"C2_1: {C2_1} \nAtoms: {atoms} \n{unmoved_atoms_count(C2_1, atoms)}")
        group_mat = np.array([unmoved_atoms_count(E_1, atoms),unmoved_atoms_count(C2_1, atoms),unmoved_atoms_count(Sv_1, atoms),unmoved_atoms_count(sv_1, atoms)])

    elif pg == "Td":
        E,E_1=identity_matrix(atoms)
        C3,C3_1=c3_matrix_z(atoms)
        C2,C2_1=c2_x(atoms)
        S4,S4_1=s_4(atoms)
        sd1,sd1_1=sigmad1(atoms)
        group_mat = np.array([unmoved_atoms_count(E_1, atoms),unmoved_atoms_count(C3_1, atoms),unmoved_atoms_count(C2_1, atoms),unmoved_atoms_count(S4_1, atoms),unmoved_atoms_count(sd1_1, atoms)])

    elif pg == "C3v":
        E,E_1=identity_matrix(atoms)
        C3,C3_1=c3_matrix_z(atoms)
        Sv,Sv_1=sigmav_xz(atoms)
        group_mat = np.array([unmoved_atoms_count(E_1, atoms),unmoved_atoms_count(C3_1, atoms),unmoved_atoms_count(Sv_1, atoms)])

    elif pg == "C4v":
        E,E_1=identity_matrix(atoms)
        C4,C4_1=c4_matrix(atoms)
        C2,C2_1=c2_matrix_z(atoms)
        sv,sv_1=sigmav(atoms)
        sd,sd_1=sigmad(atoms)
        group_mat = np.array([unmoved_atoms_count(E_1, atoms),unmoved_atoms_count(C4_1, atoms),unmoved_atoms_count(C2_1, atoms),unmoved_atoms_count(sv_1, atoms),unmoved_atoms_count(sd_1, atoms)])

    elif pg == "D4h":
        E,E_1=identity_matrix(atoms)
        C4,C4_1=c4_matrix(atoms)
        C2,C2_1=c2_matrix_z(atoms)
        C21,C21_1=c2_x(atoms)
        c22,c22_1=c2_dprime(atoms)
        I,I_1= inversion(atoms)
        S4,S4_1=s4(atoms)
        sh,sh_1=sigmah(atoms)
        sv,sv_1=sigmav(atoms)
        sd,sd_1=sigmad(atoms)
        group_mat = np.array([unmoved_atoms_count(E_1, atoms),unmoved_atoms_count(C4_1, atoms),unmoved_atoms_count(C2_1, atoms),unmoved_atoms_count(C21_1, atoms),unmoved_atoms_count(c22_1, atoms),unmoved_atoms_count(I_1, atoms),unmoved_atoms_count(S4_1, atoms),unmoved_atoms_count(sh_1, atoms),unmoved_atoms_count(sv_1, atoms),unmoved_atoms_count(sd_1, atoms)])
        
    else:
        print("ERROR")
        group_mat = np.array([-1])
    
    return group_mat

def get_data_from_point_group(pg_, df_):
    """returns the relevant data (symmetry coefficients, order, dataframe) from the point group argument"""
    searchfor=['x','y','z']
    symmetry_op_coeff = df_.iloc[-2,1:].dropna().to_numpy()
    symmetry_op_coeff = symmetry_op_coeff.flatten() #convert to 1d array
    
    
    atomic_contribution_symm = df_.iloc[-1,1:].dropna().to_numpy()
    atomic_contribution_symm = atomic_contribution_symm.flatten() #convert to 1d array
    
    order =np.sum(symmetry_op_coeff) # This gives the order of the undergrad group
    return symmetry_op_coeff, order, atomic_contribution_symm 

def calculate_irreducible_representations(df_, order_, unmoved_atoms, atomic_contribution_symm, symmetry_coefficients):
  M=[]
  rows=len(df_.index)-1
  for i in range(rows):
    N1=0
    row_list_all=df_.loc[i].to_list()
    row_list=row_list_all[1:-3]
    for j in range(len(row_list)):
        N=(unmoved_atoms[j]*symmetry_coefficients[j]*row_list[j]*atomic_contribution_symm[j])
        N1 += N
    N2=N1/order_
    M.append(N2)
  M.append(0)
  df_['M']=M #the coefficient of the particular irreducible representation present
  irreducible_rep = df_[df_.M > 0]
  return irreducible_rep

def gamma_formula_notation(df_):
  coefficients = df_["M"]
  symbols = df_.iloc[:, 0]
  
  # Initialize an empty list to store the terms
  terms = []
  
  # Iterate over the symbols and coefficients
  for symbol, coefficient in zip(symbols, coefficients):
      # Check if the coefficient is non-zero
      if coefficient != 0:
          # Construct the term as "coefficient symbol"
          term = f"${coefficient}$ {symbol}"
          terms.append(term)
  
  # Join the terms with " + " separator
  sum_string = " + ".join(terms)
  
  return sum_string