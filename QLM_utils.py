# -*- coding:utf-8 -*- 
from __future__ import division
import numpy as np 
import os
import math
import pickle as Pickle
import pynlpir 
import random
from math import log
from numpy import linalg as LA

def F(rhoM, proDict): 
	# print rhoM
	res = 0
	for pm in proDict:
		P = np.trace(np.dot(proDict[pm][1], rhoM))
		res += proDict[pm][0] * log(P)
		# print('value of target F function = {}'.format(res))
	return res

def Grad_F(rhoM, proDict, dim): 
	res = np.zeros((dim, dim))
	for pm in  proDict:
		trace_val = np.trace(np.dot(proDict[pm][1], rhoM))
		res += (proDict[pm][0] * proDict[pm][1] / trace_val)
	return res

def rho_bar(rhoM, proDict, dim):
	grad_f = Grad_F(rhoM, proDict, dim)
	res = (np.dot(grad_f,rhoM) + np.dot(rhoM,grad_f))/2
	return res


def rho_tilde(rhoM, proDict, dim):
	grad_f = Grad_F(rhoM, proDict, dim)
	grad_rho_grad = np.dot(np.dot(grad_f, rhoM),grad_f)
	res = grad_rho_grad/np.trace(grad_rho_grad)
	return res

def D_bar(rhoM, proDict, dim):
	return(rho_bar(rhoM, proDict,dim)-rhoM)

def D_tilde(rhoM, proDict, dim):
	return(rho_tilde(rhoM, proDict,dim)-rhoM)


def q_t(t, rhoM, proDict, dim):
	grad_f = Grad_F(rhoM, proDict, dim)
	grad_rho_grad = np.dot(np.dot(grad_f, rhoM),grad_f)
	res = 1+2*t+t*t*np.trace(grad_rho_grad)
	return res


def D(t, rhoM, proDict, dim): # 公式（19）
	grad_f = Grad_F(rhoM, proDict, dim)
	grad_rho_grad = np.dot(np.dot(grad_f, rhoM),grad_f)
	d_bar = D_bar(rhoM, proDict, dim)
	d_tilde = D_tilde(rhoM, proDict,dim)
	q = q_t(t, rhoM, proDict, dim)
	res = (2/q)*d_bar + (t*np.trace(grad_rho_grad)/q)*d_tilde
	return res


def set_t(t):
	return max(1, t)


def judgement(rhoM, proDict,f_old, dim, threshold_values = (1e-7, 1e-7, 1e-7)): 
	grad_f = Grad_F(rhoM, proDict, dim)
	grad_rho_grad = np.dot(np.dot(grad_f, rhoM),grad_f)
	grad_rho = np.dot(grad_f,rhoM)
	diff = f_old - F(rhoM, proDict)

	if(LA.norm(rhoM - grad_rho_grad)< threshold_values[0] and LA.norm(rhoM -grad_rho)< threshold_values[1] and abs(diff)< threshold_values[2]):
		return False
	else:
		return True


def judge_t(t, d, rhoM, proDict, dim, iter_r):
	# print 'please see here:'
	f_new = F(rhoM + t * d, proDict)
	f_old = F(rhoM, proDict)
	diff = iter_r * t * np.trace(np.dot(Grad_F(rhoM, proDict,dim), d))
	if(f_new == f_old):
		return False
	# print(f_new-f_old)
	return(f_new <=f_old+diff)


def test_set_generator(proj_num,vector_dim):
	dictionary = {}
	for i in range(proj_num):
		weight = np.random.random()
		vector = np.random.rand(1,vector_dim)
		vector = vector/(math.sqrt(np.dot(vector,np.transpose(vector))))
		projector = np.outer(vector, vector) / np.inner(vector, vector)
		dictionary['word_'+str(i)] = [weight, projector]

	return dictionary