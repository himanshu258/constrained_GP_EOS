import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import minimize
# from scipy.optimize import Bounds
from scipy.linalg import cho_solve
# from pyDOE import lhs
import scipy.optimize as opt
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from sklearn.metrics import mean_squared_error
from scipy.linalg import cho_solve
import pandas as pd

def test(X, n_test):
    X1 = np.linspace(X[:,0].max(), X[:,0].min(), n_test)
    X2 = np.linspace(X[:,1].min(), X[:,1].max(), n_test)
    X1P, X2P = np.meshgrid(X1, X2)
    X1E, X2E = np.meshgrid(X2, X1)
    X_testP = np.hstack((X1P.reshape(-1,1), X2P.reshape(-1,1)))
    X_testE = np.hstack((X2E.reshape(-1,1), X1E.reshape(-1,1)))
    return X_testP, X_testE


def poly_regression(x_train, y_train, x_test, n_test, degree):
    
    poly = PolynomialFeatures(degree)
    
    y_P = y_train[:x_train.shape[0]]
    y_E = y_train[x_train.shape[0]:]
    
    poly_train = poly.fit_transform(x_train)
    poly_test = poly.fit_transform(x_test)
    
    reg_P = LinearRegression().fit(poly_train, y_P)
    reg_E = LinearRegression().fit(poly_train, y_E)
    
    mu_P = reg_P.predict(poly_train)
    mu_E = reg_E.predict(poly_train)
    
    mu_test_P = reg_P.predict(poly_test)
    mu_test_E = reg_E.predict(poly_test)
    
    mu_regression = np.concatenate((mu_P,mu_E))      
    mu_test_regression = np.concatenate((mu_test_P,mu_test_E))
    
    return mu_regression, mu_test_regression

def gradientFunction(x_train, y_train, n_test, degree):
    
    X_testP, X_testE = test(x_train, n_test)
    _ , mu_test_regression_P = poly_regression(x_train, y_train, X_testP,n_test, degree)
    _, mu_test_regression_E = poly_regression(x_train, y_train, X_testE,n_test, degree)
    
    
    y_pred_P = np.squeeze(mu_test_regression_P[:X_testP.shape[0]])
    y_pred_E = np.squeeze(mu_test_regression_E[X_testE.shape[0]:])
    gradP_V = np.gradient(y_pred_P, X_testP[:,0])
    gradE_T = np.gradient(y_pred_E, X_testE[:,1])
    
    return gradP_V, gradE_T

def poly_regression_prime(x_train, y_train, x_virtual, n_test, degree):
    
    gradP_V, gradE_T = gradientFunction(x_train, y_train, n_test, degree)
    X_testP, X_testE = test(x_train, n_test)
    
    poly = PolynomialFeatures(degree)
    
    y_P = gradP_V.reshape(-1,1)
    y_E = gradE_T.reshape(-1,1)
    
    poly_train_P = poly.fit_transform(X_testP)
    poly_train_E = poly.fit_transform(X_testE)
    
    poly_test = poly.fit_transform(x_virtual)
    
    reg_P = LinearRegression().fit(poly_train_P, y_P)
    reg_E = LinearRegression().fit(poly_train_E, y_E)
    
    
    mu_virtual_P = reg_P.predict(poly_test)
    mu_virtual_E = reg_E.predict(poly_test)
    
         
    mu_virtual_regression = np.concatenate((mu_virtual_P,mu_virtual_E))
    
    return mu_virtual_regression


def Kernel(x1, x2, l1, l2, sigma_f, sigma_n1, sigma_n2, constraints=1):
    
    theta = np.array([1/(2*l1**2), 1/(2*l2**2)])
    diff_x = x1[:, None, :] - x2[None, :, :]
    sq_diff_x = diff_x ** 2
    exp_term = np.exp(-np.sum(theta * sq_diff_x, axis=2))

    K11 = sigma_f**2 * exp_term * (2*theta[0] - 4*theta[0]**2 * sq_diff_x[:,:,0]) + np.eye(x1.shape[0],x2.shape[0]) * (sigma_n1)**2
    K22 = sigma_f**2 * (exp_term +
                      x1[:, 1][:, None] * exp_term * (2 * theta[1] * (x1[:, 1][:, None] - x2[:, 1][None, :])) +
                      x2[:, 1][None, :] * exp_term * (2 * theta[1] * (-x1[:, 1][:, None] + x2[:, 1][None, :])) +
                      x1[:, 1][:, None] * x2[:, 1][None, :] * exp_term * (2 * theta[1] - 4 * theta[1]**2 * sq_diff_x[:,:,1])) + np.eye(x1.shape[0],x2.shape[0]) * (sigma_n2)**2
    K21 = sigma_f**2 * (np.exp(-np.sum(theta * sq_diff_x, axis=2)) * (2 * theta[0] * (-x1[:, 0][:, None] + x2[:, 0][None, :])) -
                      x1[:, 1][:, None] * np.exp(-np.sum(theta * sq_diff_x, axis=2)) * (4 * theta[0] * theta[1] * (x1[:, 0][:, None] - x2[:, 0][None, :]) * (x1[:, 1][:, None] - x2[:, 1][None, :])))
    K12 = sigma_f**2 * (np.exp(-np.sum(theta * sq_diff_x, axis=2)) * (2 * theta[0] * (x1[:, 0][:, None] - x2[:, 0][None, :])) -
                      x2[:, 1][None, :] * np.exp(-np.sum(theta * sq_diff_x, axis=2)) * (4 * theta[0] * theta[1] * (x1[:, 0][:, None] - x2[:, 0][None, :]) * (x1[:, 1][:, None] - x2[:, 1][None, :])))

    K_block = np.block([[K11,constraints*K12],[constraints*K21,K22]])
    

    return K_block, K11, K12, K21, K22


def Kernel_prime(x1, x2, l1, l2, sigma_f, sigma_n1, sigma_n2, constraints=1):
    theta = np.array([1/(2*l1**2), 1/(2*l2**2)])

    diff_x = x1[:, None, :] - x2[None, :, :]
    sq_diff_x = diff_x ** 2

    K11prime = sigma_f**2 * np.exp(-np.sum(theta * sq_diff_x, axis=2)) * (4 * theta[0]**2 * (-8 * theta[0] * sq_diff_x[:,:,0] + (2 * theta[0] * sq_diff_x[:,:,0] - 1)**2 + 2)) + + np.eye(x1.shape[0],x2.shape[0]) * (sigma_n1)**2
    K12prime = sigma_f**2 * np.exp(-np.sum(theta * sq_diff_x, axis=2)) * (4 * x2[:, 1][None, :] * theta[0] * theta[1] * (2 * theta[0] * sq_diff_x[:,:,0] - 1) * (2 * theta[1] * sq_diff_x[:,:,1] - 1))
    K21prime = sigma_f**2 * np.exp(-np.sum(theta * sq_diff_x, axis=2)) * (4 * x1[:, 1][:, None] * theta[0] * theta[1] * (2 * theta[0] * sq_diff_x[:,:,0] - 1) * (2 * theta[1] * sq_diff_x[:,:,1] - 1))
    K22prime = sigma_f**2 * np.exp(-np.sum(theta * sq_diff_x, axis=2)) * (4 * x1[:, 1][:, None] * x2[:, 1][None, :] * theta[1]**2 * (-8 * theta[1] * sq_diff_x[:,:,1] + (2 * theta[1] * sq_diff_x[:,:,1] - 1)**2 + 2)) + np.eye(x1.shape[0],x2.shape[0]) * (sigma_n2)**2
    K_prime = np.block([[K11prime,constraints*K12prime],[constraints*K21prime,K22prime]])
    
    return K_prime, K11prime, K12prime, K21prime, K22prime


def mle(theta, x_train, y_train, mean_regression, constraints=1):
    
    l1,l2, sigma_f, sigma_n1, sigma_n2 = theta  
    n = x_train.shape[0]      
    K = Kernel(x_train, x_train, 10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2, constraints)[0]   + np.eye(2*n)*1e-10
    L = np.linalg.cholesky(K)
    var = ((y_train - mean_regression).T @ (cho_solve((L, True), y_train - mean_regression))) 
    LnofDetK = 2*np.sum(np.log(np.abs(np.diag(L)))) 
    le = -(1/2)*var - 0.5*LnofDetK - (2*n/2)*np.log(2*np.pi)
    
    return -le.flatten()


def predict(theta, x_train, y_train, x_test, n_test, degree, constraints=1):
    
    l1, l2, sigma_f, sigma_n1, sigma_n2 = np.power(10, theta)
    n = x_train.shape[0]
    
    mu_reg, mu_test_reg = poly_regression(x_train, y_train,x_test,  n_test, degree)
    
    k12 = Kernel(x_train, x_test, l1, l2, sigma_f, sigma_n1, sigma_n2, constraints)[0]
    k21 = Kernel(x_test, x_train, l1, l2, sigma_f, sigma_n1, sigma_n2, constraints)[0]
    k22 = Kernel(x_test, x_test, l1, l2, sigma_f, sigma_n1, sigma_n2, constraints)[0]
    K = Kernel(x_train, x_train, l1, l2, sigma_f, sigma_n1, sigma_n2, constraints)[0] + np.eye(2 * n) * 1e-10
    L = np.linalg.cholesky(K)

    alpha = cho_solve((L, True), y_train - mu_reg)
    y_gp = mu_test_reg + k21 @ alpha
    varr = k22 - k21 @ cho_solve((L, True), k12)
    pred_var = np.diag(varr)
    pred_std = np.sqrt(pred_var)

    return y_gp.flatten(), pred_std.flatten()
    
def predict_prime(theta, x_train, y_train, x_test, n_test, degree, constraints=1):
    
        l1,l2, sigma_f, sigma_n1, sigma_n2 = theta
        n = x_train.shape[0]
        y_traind = poly_regression_prime(x_train,y_train, x_train, n_test, degree)
        mu_test_reg = poly_regression_prime(x_train,y_train, x_test,n_test, degree)
        mu_reg = poly_regression_prime(x_train,y_train, x_train, n_test, degree)
        
        k12 = Kernel_prime(x_train, x_test, 10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2, constraints)[0]  
        k21 = Kernel_prime(x_test, x_train, 10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2, constraints)[0]
        k22 = Kernel_prime(x_test, x_test, 10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2, constraints)[0] 
        K = Kernel_prime(x_train, x_train, 10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2, constraints)[0] +  np.eye(2*n)*1e-10
        L = np.linalg.cholesky(K)
        y_gp = mu_test_reg + k21 @ (cho_solve((L, True), y_traind - mu_reg))
        varr = k22 - k21 @ (cho_solve((L, True), k12))
        pred_var = varr.diagonal()
        pred_var1 = pred_var.flatten()
        pred_var1[pred_var1<0.0]=0.0

        return y_gp.flatten(), pred_var1


def Kblock_unified(x1,x2,x_test,l1,l2,sigma_f, sigma_n1, sigma_n2):
    
    Kblock_unified_block = np.block([[Kernel(x1,x1,l1,l2,sigma_f, sigma_n1, sigma_n2, constraints=1)[1],Kernel(x1,x2,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[2]],[Kernel(x2,x1,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[3],Kernel(x2,x2,l1,l2,sigma_f, sigma_n1, sigma_n2, constraints=1)[4]]])
    Kblock_unified_block12 = np.block([[Kernel(x1,x_test,l1,l2,sigma_f, sigma_n1, sigma_n2, constraints=1)[1],Kernel(x1,x_test,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[2]],[Kernel(x2,x_test,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[3],Kernel(x2,x_test,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[4]]])
    Kblock_unified_block21 =  np.block([[Kernel(x_test,x1,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[1],Kernel(x_test,x2,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[2]],[Kernel(x_test,x1,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[3],Kernel(x_test,x2,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[4]]])
    
    return [Kblock_unified_block, Kblock_unified_block12, Kblock_unified_block21]

def Kblock_unified_prime(x1,x2,x_test,l1,l2,sigma_f, sigma_n1, sigma_n2):
    
    Kblock_unified_block_prime = np.block([[Kernel_prime(x1,x1,l1,l2,sigma_f, sigma_n1, sigma_n2, constraints=1)[1],Kernel_prime(x1,x2,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[2]],[Kernel_prime(x2,x1,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[3],Kernel_prime(x2,x2,l1,l2,sigma_f, sigma_n1, sigma_n2, constraints=1)[4]]])
    Kblock_unified_block12_prime = np.block([[Kernel_prime(x1,x_test,l1,l2,sigma_f, sigma_n1, sigma_n2, constraints=1)[1],Kernel_prime(x1,x_test,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[2]],[Kernel_prime(x2,x_test,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[3],Kernel_prime(x2,x_test,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[4]]])
    Kblock_unified_block21_prime =  np.block([[Kernel_prime(x_test,x1,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[1],Kernel_prime(x_test,x2,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[2]],[Kernel_prime(x_test,x1,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[3],Kernel_prime(x_test,x2,l1,l2,sigma_f,sigma_n1, sigma_n2, constraints=1)[4]]])
    
    return [Kblock_unified_block_prime, Kblock_unified_block12_prime, Kblock_unified_block21_prime]

def poly_regression_unified(x_train1, x_train2, y_train, x_test, degree):
    
    poly = PolynomialFeatures(degree)
    
    y_P = y_train[:x_train1.shape[0]]
    y_E = y_train[x_train1.shape[0]:]
    
    poly_train_P = poly.fit_transform(x_train1)
    poly_train_E = poly.fit_transform(x_train2)
    poly_test = poly.fit_transform(x_test)
    
    reg_P = LinearRegression().fit(poly_train_P, y_P)
    reg_E = LinearRegression().fit(poly_train_E, y_E)
    
    mu_P = reg_P.predict(poly_train_P)
    mu_E = reg_E.predict(poly_train_E)
    
    mu_test_P = reg_P.predict(poly_test)
    mu_test_E = reg_E.predict(poly_test)
    
    mu_regression = np.concatenate((mu_P,mu_E))      
    mu_test_regression = np.concatenate((mu_test_P,mu_test_E))
    
    return mu_regression, mu_test_regression

def gradientFunction(x_train1, x_train2, y_train, degree,n_test):
    
    X_testP, X_testE = test(x_train1, n_test)
    _ , mu_test_regression_P = poly_regression_unified(x_train1, x_train2, y_train, X_testP, degree)
    _, mu_test_regression_E = poly_regression_unified(x_train1, x_train2, y_train, X_testE, degree)
    
    
    y_pred_P = np.squeeze(mu_test_regression_P[:X_testP.shape[0]])
    y_pred_E = np.squeeze(mu_test_regression_E[X_testE.shape[0]:])
    gradP_V = np.gradient(y_pred_P, X_testP[:,0])
    gradE_T = np.gradient(y_pred_E, X_testE[:,1])
    
    return gradP_V, gradE_T

def poly_regression_prime_unified(x_train1, x_train2, y, x_virtual, degree):
    
    gradP_V, gradE_T = gradientFunction(x_train1, x_train2, y,degree, n_test = 30)
    X_testP, X_testE = test(x_train1, n_test= 30)
    
    poly = PolynomialFeatures(degree)
    
    y_P_prime = gradP_V.reshape(-1,1)
    y_E_prime = gradE_T.reshape(-1,1)
    
    poly_train_P = poly.fit_transform(X_testP)
    poly_train_E = poly.fit_transform(X_testE)
    
    poly_test = poly.fit_transform(x_virtual)
    
    reg_P = LinearRegression().fit(poly_train_P, y_P_prime)
    reg_E = LinearRegression().fit(poly_train_E, y_E_prime)
    
    
    mu_virtual_P = reg_P.predict(poly_test)
    mu_virtual_E = reg_E.predict(poly_test)
    
         
    mu_virtual_regression = np.concatenate((mu_virtual_P,mu_virtual_E))
    
    return mu_virtual_regression


def mle_unified(theta, x_train1, x_train2, y_train, mean_regression):
    
    l1,l2, sigma_f, sigma_n1, sigma_n2 = theta  
    n1 = x_train1.shape[0]
    n2 = x_train2.shape[0]      
    K = Kblock_unified(x_train1, x_train2,x_train2, 10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2)[0]   + np.eye(n1+n2)*1e-10
    L = np.linalg.cholesky(K)
    var = ((y_train - mean_regression).T @ (cho_solve((L, True), y_train - mean_regression))) 
    LnofDetK = 2*np.sum(np.log(np.abs(np.diag(L)))) 
    le = -(1/2)*var - 0.5*LnofDetK - ((n1+n2)/2)*np.log(2*np.pi)
    
    return -le.flatten()

def predict_unified(theta, x_train1, x_train2, y_train, x_test, n_test, degree, constraints=1):
    
    
    l1,l2, sigma_f, sigma_n1, sigma_n2 = theta
    n1 = x_train1.shape[0]
    n2 = x_train2.shape[0]
    mu_reg, mu_test_reg = poly_regression_unified(x_train1, x_train2, y_train, x_test, degree)
        
    k12 = Kblock_unified(x_train1,x_train2, x_test, 10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2)[1]  
    k21 = Kblock_unified(x_train1,x_train2, x_test, 10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2)[2]
    k22 = Kblock_unified(x_test, x_test,x_test, 10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2)[0] 
    K = Kblock_unified(x_train1, x_train2,x_test, 10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2)[0] +  np.eye(n1+n2)*1e-10
    L = np.linalg.cholesky(K)

    y_gp = mu_test_reg + k21 @ (cho_solve((L, True), y_train - mu_reg))
    varr = k22 - k21 @ (cho_solve((L, True), k12))
    pred_var = varr.diagonal()
    pred_std = np.sqrt(pred_var)

      
    return y_gp.flatten(), pred_std.flatten()

def predict_prime_unified(theta, x_train1, x_train2, y_train, x_test,n_test, degree, constraints=1):
    
        l1,l2, sigma_f, sigma_n1, sigma_n2 = theta
        n1 = x_train1.shape[0]
        n2 = x_train2.shape[0]
        
        mu_test_reg = poly_regression_prime_unified(x_train1, x_train2,y_train, x_test, degree)
        
        k12 = Kblock_unified_prime(x_train1,x_train2, x_test, 10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2) [1] 
        k21 = Kblock_unified_prime(x_train1,x_train2, x_test, 10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2)[2]
        k22 = Kblock_unified_prime(x_test, x_test, x_test,10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2)[0] 
        K = Kblock_unified_prime(x_train1, x_train2, x_test,10**l1, 10**l2, 10**sigma_f, 10**sigma_n1, 10**sigma_n2)[0] +  np.eye(n1+n2)*1e-10
        L = np.linalg.cholesky(K)

        y_gp = mu_test_reg 
        varr = k22 - k21 @ (cho_solve((L, True), k12))
        pred_var = varr.diagonal()
        pred_var1 = pred_var.flatten()
        pred_var1[pred_var1<0.0]=0.0

      
        return y_gp.flatten(), pred_var1
    
    