import numpy as np
import torch
from scipy.stats import qmc

class Pendulum_on_a_cart:
    def __init__(self,ode_args):
        self.m = ode_args['m']
        self.l = ode_args['l']
        self.M = ode_args['M']
        self.g = ode_args['g']
        
        self.invariant_state = 2
        self.nbr_states = 4
        self.dimq = int(self.nbr_states/2)
        
        self.alpha = self.m* self.l**2
        self.beta = self.m*self.l
        self.gamma = self.M+self.m
        self.D = -self.m*self.g*self.l

        self.sampler = qmc.Halton(d=self.nbr_states)
        self.symmGrid = self.generate_grid()

        self.translation_factor = np.array([[1,0]])
        self.rotation_factor =  np.array([[[0,0],[0,0]]])
        
    def ode(self,t,x):
        q = x[:2]
        p = x[2:]
        sinT = np.sin(q[1])
        cosT = np.cos(q[1])
               
        dx = np.zeros(x.shape)
        dx[0] = 1/(self.alpha*self.gamma-self.beta**2*cosT**2) * (self.alpha*p[0] - self.beta*cosT*p[1])
        dx[1] = 1/(self.alpha*self.gamma-self.beta**2*cosT**2) * (-self.beta*cosT*p[0] + self.gamma*p[1])
        
        dx[2] = 0
        dx[3] = (-self.beta*p[0]*p[1]*sinT/(self.alpha*self.gamma-self.beta**2*cosT**2) 
        + ( self.beta**2*sinT*cosT*(p[0]**2*self.alpha-2*self.beta*cosT*p[0]*p[1]+self.gamma*p[1]**2) )/((self.alpha*self.gamma-self.beta**2*cosT**2)**2)    
        - self.D*sinT)
        
        return dx #q_dot, p_dot = dH/dp, -dH/dq
    
    def hamiltonian(self,x):
        q = x[:2,:]
        p = x[2:,:]
        
        cosT = np.cos(q[1,:])
        H = 1/2 * 1/(self.alpha*self.gamma-self.beta**2*cosT**2) * (self.alpha*p[0,:]**2 - 2*self.beta*cosT*p[0,:]*p[1,:] + self.gamma*p[1,:]**2) -self.D*cosT
        return H
    
    def get_kartesian_coordinates(self,x):
        x_cart = x[:,0]
        x_pend = x_cart + self.l*np.sin(x[:,1])
        y_pend = self.l*np.cos(x[:,1])
        return x_cart, x_pend, y_pend
    
    def generate_random_inital_value(self, quantity=1, region_tag=None):
        if (region_tag=='Evaluation'):
            upper_bound = [ 10, 2*np.pi, 2, 2*np.pi]
            lower_bound = [-10,-2*np.pi,-2,-2*np.pi]
        else:
            upper_bound = [ 5, np.pi, 1, np.pi]
            lower_bound = [-5,-np.pi,-1,-np.pi]
        samples = self.sampler.random(n=quantity)
        samples = qmc.scale(samples,lower_bound, upper_bound)

        if quantity==1:
            samples = samples.flatten()

        return samples

    def generate_grid(self):
        upper_bound = [ 5, np.pi, 1, np.pi]
        lower_bound = [-5,-np.pi,-1,-np.pi]
        samples = self.sampler.random(n=81)
        samples = qmc.scale(samples,lower_bound, upper_bound)
        samples = torch.Tensor(samples)
        samples.requires_grad = True

        return samples


class Kepler_cartesian:
    def __init__(self,ode_args=None):
        self.nbr_states = 4 
        self.dimq = int(self.nbr_states/2)

        self.m = ode_args['m'] 
        self.M_inv = 1/self.m * np.identity(self.dimq)
        self.g = ode_args['g'] 
        self.k = 1.016895192894334*1e3 #g*m*M

        self.sampler = qmc.Halton(d=self.nbr_states)
        self.symmGrid = self.generate_grid()

        self.translation_factor = np.array([[0,0]])
        self.rotation_factor = np.array([[[0,np.sqrt(2)/2.],[-np.sqrt(2)/2.,0]]])
    
    def ode(self, t, x):
        if len(x.shape)==1:
            x = np.expand_dims(x,axis=-1)
        q = x[:self.dimq,:]
        p = x[self.dimq:,:]

        dx = np.zeros(x.shape)
        dx[:self.dimq] = self.M_inv @ p
        dx[self.dimq:] = -self.k*q/(np.linalg.norm(q)**3)

        dx = dx.squeeze()
        return dx

    def hamiltonian(self,x):
        q = x[:self.dimq,:]
        p = x[self.dimq:,:]
        return  np.array([1/2 * p[:,i].T @ self.M_inv @ p[:,i]  - self.k/np.linalg.norm(q[:,i]) for i in range(p.shape[-1])])


    def generate_random_inital_value(self,quantity=1,region_tag=None):
        states = self.sampler.random(n=quantity)
        if region_tag=='Evaluation':
            scaled_states = qmc.scale(states[:,:3], [-10.0,-10.0,5.0], [10.0,10.0,20])     #x,y,radius
            radius = np.expand_dims(scaled_states[:,2], axis=-1)
        else:
            scaled_states = qmc.scale(states[:,:3], [-10.0,-10.0,5.0], [10.0,10.0,10])     #x,y,radius
            radius = np.expand_dims(scaled_states[:,2], axis=-1)

        q = np.multiply( radius/np.expand_dims(np.linalg.norm(scaled_states[:,:2],axis=1),axis=-1),scaled_states[:,:2]) #qmc.scale(states[:,:2], lower_bound, upper_bound) #np.stack([.squeeze()],axis=1)
        random_sign = np.array([(2*np.random.randint(0,2,size=(quantity))-1)]*2).T
        p = random_sign*np.stack([np.ones((quantity,)), -scaled_states[:,0]/scaled_states[:,1] ],axis=1)        #orthogonal vector
        perfect_p = np.expand_dims(np.sqrt(self.k/np.linalg.norm(q,axis=1)),axis=-1)
        p *= np.multiply( perfect_p+(0.3*perfect_p*(2*np.random.rand(quantity,1)-1)),np.expand_dims(1/np.linalg.norm(p,axis=1),axis=-1))     #scale to ||p|| = sqrt(k/||q||) (möglichst Kreise)

        init_value = np.hstack([q,p])

        if quantity==1:
            init_value = init_value.flatten()
        return init_value

    def generate_grid(self):
        samples = self.generate_random_inital_value(81)
        samples = torch.Tensor(samples)
        samples.requires_grad = True

        return samples