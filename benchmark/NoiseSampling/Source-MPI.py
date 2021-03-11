from mpi4py import MPI
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL) # terminate program with Ctrl-C

from qulacs import ParametricQuantumCircuit
import scipy.optimize

import matplotlib.pyplot as plt
import numpy as np
import time 
import random
import scipy.linalg

from qulacs import QuantumState
from qulacs import QuantumCircuit
from qulacs.gate import DenseMatrix
from qulacs.circuit import QuantumCircuitOptimizer

from qulacs import QuantumState
#固定ゲート
from qulacs.gate import Identity, X,Y,Z #パウリ演算子
from qulacs.gate import H,S,Sdag, sqrtX,sqrtXdag,sqrtY,sqrtYdag #1量子ビット Clifford演算
from qulacs.gate import T,Tdag #1量子ビット 非Clifford演算
from qulacs.gate import CNOT, CZ, SWAP #2量子ビット演算

#パラメータ付きゲート
from qulacs.gate import RX,RY,RZ #パウリ演算子についての回転演算
from qulacs.gate import U1,U2,U3 #IBM Gate
from qulacs.gate import ParametricPauliRotation
from qulacs.gate import PauliRotation

#ノイズ関連
from qulacs.gate import DepolarizingNoise,TwoQubitDepolarizingNoise
from qulacs import NoiseSimulatorMPI

from qulacs import Observable

def show_observable(hamiltonian):
    for j in range(hamiltonian.get_term_count()):
        pauli=hamiltonian.get_term(j)

        # Get the subscript of each pauli symbol
        index_list = pauli.get_index_list()

        # Get pauli symbols (I,X,Y,Z -> 0,1,2,3)
        pauli_id_list = pauli.get_pauli_id_list()

        # Get pauli coefficient
        coef = pauli.get_coef()

        # Create a copy of pauli operator
        another_pauli = pauli.copy()

        s = ["I","X","Y","Z"]
        pauli_str = [s[i] for i in pauli_id_list]
        terms_str = [item[0]+str(item[1]) for item in zip(pauli_str,index_list)]
        full_str = str(coef) + " " + " ".join(terms_str)
        print(full_str)

def define_Ising_Hamiltonian(operator,ListOfInt):
    nqubits = operator.get_qubit_count()
    
    for k in range(len(ListOfInt)):
        operator.add_operator(ListOfInt[k][2],"Z {0}".format(ListOfInt[k][0])+"Z {0}".format(ListOfInt[k][1]))
    return operator

#definition of interaction pattern of Ising model

def define_Jij(l_x,l_y):
    Jij = []

    for i in range(l_x -1):
        for j in range(l_y):
            #x direction
            Jij.append([i+j*l_x,i+1+j*l_x,1-2*random.random()]) 

    for i in range(l_x ):
        for j in range(l_y -1 ):        
            # y direction
            Jij.append([i+j*l_x,i+(j+1)*l_x,1-2*random.random()]) 
    return Jij

def define_Heisenberg_Hamiltonian(operator,ListOfInt):
    nqubits = operator.get_qubit_count()
    for k in range(len(ListOfInt)):
        operator.add_operator(1.0,"Z {0}".format(ListOfInt[k][0])+"Z {0}".format(ListOfInt[k][1]))
        operator.add_operator(1.0,"X {0}".format(ListOfInt[k][0])+"X {0}".format(ListOfInt[k][1])) 
        operator.add_operator(1.0,"Y {0}".format(ListOfInt[k][0])+"Y {0}".format(ListOfInt[k][1])) 
    return operator




def define_Z_field(operator):
    nqubits = operator.get_qubit_count()
    for k in range(nqubits):
        operator.add_operator(1.0,"Z {0}".format(k)) 
    return operator

def define_X_field(operator):
    nqubits = operator.get_qubit_count()
    for k in range(nqubits):
        operator.add_operator(1.0,"X {0}".format(k)) 
    return operator

def NoiseGate(index_list, noise_prob):
    if len(index_list) == 1:
        return DepolarizingNoise(index_list[0],noise_prob)
    elif len(index_list) == 2:
        return TwoQubitDepolarizingNoise(index_list[0],index_list[1],noise_prob)
    else:
        raise RuntimeError("")

def hamiltonian_ansatz(hamiltonian,driver,max_depth, prob):
    i=0
    nqubits = hamiltonian.get_qubit_count()
    ansatz_circuit =ParametricQuantumCircuit(nqubits)

    for depth in range(max_depth):
        for j in range(hamiltonian.get_term_count()):
            pauli = hamiltonian.get_term(j)

            # Get the subscript of each pauli symbol
            index_list = pauli.get_index_list()

            # Get pauli symbols (I,X,Y,Z -> 0,1,2,3)
            pauli_id_list = pauli.get_pauli_id_list()

            ansatz_circuit.add_parametric_gate(ParametricPauliRotation(index_list, pauli_id_list, np.pi*random.random()))
            ansatz_circuit.add_gate(NoiseGate(index_list,prob))
            i+=1

        for j in range(driver.get_term_count()):
            pauli = driver.get_term(j)
            # Get the subscript of each pauli symbol
            index_list = pauli.get_index_list()
            # Get pauli symbols (I,X,Y,Z -> 0,1,2,3)
            pauli_id_list = pauli.get_pauli_id_list()
            ansatz_circuit.add_parametric_gate(ParametricPauliRotation(index_list, pauli_id_list, np.pi*random.random()))
            ansatz_circuit.add_gate(NoiseGate(index_list,prob))
            i+=1    
          
    #print(ansatz_circuit.get_parameter_count())
    #print(ansatz_circuit)
    return ansatz_circuit

def cost_func_ansatz(ansatz_circuit,hamiltonian,para):
    nqubits = ansatz_circuit.get_qubit_count()
    state = QuantumState(nqubits)

    parameter_count = ansatz_circuit.get_parameter_count()



    for i in range(parameter_count):
        ansatz_circuit.set_parameter(i,para[i])
        
    ansatz_circuit.update_quantum_state(state)

    return  hamiltonian.get_expectation_value(state)



l_x = 4
l_y = 4
max_depth = 3
sample_number = 10000
prob = 0.01

nqubits = l_x * l_y

# 乱数シード値の設定
random.seed(int(1e9+7))
#測定する演算子の定義
measurements = Observable(nqubits)
define_Z_field(measurements)
show_observable(measurements)

#相互作用パターンの定義
Jij = define_Jij(l_x,l_y)
print(Jij)

#ドライバーの定義
xdriver = Observable(nqubits)
define_X_field(xdriver)

#ハミルトニアンの定義
Ising_Hamiltonian =  Observable(nqubits)
define_Ising_Hamiltonian(Ising_Hamiltonian,Jij)
show_observable(Ising_Hamiltonian)

#初期状態の定義
state = QuantumState(nqubits)
state.set_zero_state()

#ハミルトニアンからパラメトリック回路を用意
circuit = hamiltonian_ansatz(Ising_Hamiltonian,xdriver,max_depth,prob)
#ノイズの高速計算
simulator = NoiseSimulatorMPI(circuit,state)

time_sta = time.perf_counter()
A = simulator.execute(sample_number)
time_end = time.perf_counter()
print("Time on NoiseSimulator: " + str(time_end - time_sta))

#ノイズの愚直計算
time_sta = time.perf_counter()
for i in range(sample_number):
    state.set_zero_state()
    circuit.update_quantum_state(state)
    A.append(state.sampling(1))
time_end = time.perf_counter()
print("Time on Normal Sampling: " + str(time_end - time_sta))
