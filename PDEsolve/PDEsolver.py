import numpy as np
from scipy.integrate import ode

# Porous-Fisher model 1 species 

class porous_1species:
    
    def __init__(self,Lx,Ly,Nx,Ny,dt,init_rho,tmax,D,r,pmax):
        self.Lx = Lx
        self.Nx = Nx
        self.Ly = Ly
        self.Ny = Ny
        self.dt = dt
        self.dx = Lx/Nx
        self.dy = Ly/Ny
        self.tmax = tmax
        self.D = D
        self.r = r
        self.pmax = pmax
        self.chalf_x = np.linspace(-Lx,Lx,int(2*Nx+1))
        self.cspan_x = np.linspace(-Lx + self.dx/2, Lx - self.dx/2, int(2*Nx))
        self.chalf_y = np.linspace(-Ly,Ly,int(2*Ny+1))
        self.cspan_y = np.linspace(-Ly + self.dy/2, Ly - self.dy/2, int(2*Ny))
        self.X, self.Y = np.meshgrid(self.cspan_x,self.cspan_y)
        self.rho = [init_rho]
        self.rho_ex = [np.concatenate([x for x in init_rho])]
    
    def splitter(self,rho_expanded):
        return np.array(np.split(rho_expanded,2*self.Nx))
    
    def disc_entropy(self,f):
        return f # + ... for nonlocal terms
    
    def velocities_x(self,h):
        return -(h[1:,:]-h[:-1,:])/self.dx
    
    def velocities_y(self,h):
        return -(h[:,1:]-h[:,:-1])/self.dy
    
    def flux_x(self,v,g):
        zero_vec = np.zeros(v.shape)
        F = g[:-1,:]*np.maximum(v,zero_vec) + g[1:,:]*np.minimum(v,zero_vec)
        return np.concatenate([[np.zeros(len(F[0]))],F,[np.zeros(len(F[0]))]])
    
    def flux_y(self,v,g):
        zero_vec = np.zeros(v.shape)
        F = g[:,:-1]*np.maximum(v,zero_vec) + g[:,1:]*np.minimum(v,zero_vec)
        return np.transpose(np.concatenate([[np.zeros(len(F))],np.transpose(F),[np.zeros(len(F))]]))
    
    def dpdt(self,t,p_expanded):
        p = self.splitter(p_expanded)
        h = self.disc_entropy(p)
        Fx = self.flux_x(self.velocities_x(h),p)
        Fy = self.flux_y(self.velocities_y(h),p)
        diff = -self.D*(Fx[1:,:]-Fx[:-1,:])/self.dx-self.D*(Fy[:,1:]-Fy[:,:-1])/self.dy + self.r*p*(1-p/self.pmax)
        return np.concatenate([x for x in diff])
    
    
    def solve(self):
        solODE = ode(self.dpdt).set_integrator('dopri5')
        solODE.set_initial_value(self.rho_ex[0],0)
        t = 0
        while t < self.tmax:
            print('t = %.2f'% t,end = '\r')
            t += self.dt
            self.rho.append(self.splitter(solODE.integrate(t)))
        return np.array(self.rho)

# Fisher-KPP model 1 species

class fisher_1species:
    
    def __init__(self,Lx,Ly,Nx,Ny,dt,init_rho,tmax,D,r,pmax):
        self.Lx = Lx
        self.Nx = Nx
        self.Ly = Ly
        self.Ny = Ny
        self.dt = dt
        self.dx = Lx/Nx
        self.dy = Ly/Ny
        self.tmax = tmax
        self.D = D
        self.r = r
        self.pmax = pmax
        self.chalf_x = np.linspace(-Lx,Lx,int(2*Nx+1))
        self.cspan_x = np.linspace(-Lx + self.dx/2, Lx - self.dx/2, int(2*Nx))
        self.chalf_y = np.linspace(-Ly,Ly,int(2*Ny+1))
        self.cspan_y = np.linspace(-Ly + self.dy/2, Ly - self.dy/2, int(2*Ny))
        self.X, self.Y = np.meshgrid(self.cspan_x,self.cspan_y)
        self.rho = [init_rho]
        self.rho_ex = [np.concatenate([x for x in init_rho])]
    
    def splitter(self,rho_expanded):
        return np.array(np.split(rho_expanded,2*self.Nx))
    
    def disc_laplacian(self,f):
        dx, dy = self.dx, self.dy
        lap = np.zeros(f.shape)
        lap[1:-1,1:-1] = (f[2:,1:-1] - 2*f[1:-1,1:-1] + f[:-2,1:-1])/dx**2 + (f[1:-1,2:] - 2*f[1:-1,1:-1] + f[1:-1,:-2])/dy**2
        lap[0,1:-1] = (f[1,1:-1] - f[0,1:-1])/dx**2 + (f[0,2:] - 2*f[0,1:-1] + f[0,:-2])/dy**2
        lap[-1,1:-1] = (-f[-1,1:-1] + f[-2,1:-1])/dx**2 + (f[-1,2:] - 2*f[-1,1:-1] + f[-1,:-2])/dy**2
        lap[1:-1,0] = (f[2:,0] - 2*f[1:-1,0] + f[:-2,0])/dx**2 + (f[1:-1,1] - f[1:-1,0])/dy**2
        lap[1:-1,-1] = (f[2:,-1] - 2*f[1:-1,-1] + f[:-2,-1])/dx**2 +(f[1:-1,-2] - f[1:-1,-1])/dy**2
        lap[0,0] = (f[1,0] - f[0,0])/dx**2 + (f[0,1] - f[0,0])/dy**2
        lap[-1,0] = (f[-2,0] - f[-1,0])/dx**2 + (f[-1,1] - f[-1,0])/dy**2
        lap[-1,-1] = (f[-2,-1] - f[-1,-1])/dx**2 + (f[-1,-2] - f[-1,-1])/dy**2
        lap[0,-1] = (f[1,-1] - f[0,-1])/dx**2 + (f[0,-2] - f[0,-1])/dy**2
        return lap
    
    def dpdt(self,t,p_expanded):
        p = self.splitter(p_expanded)
        h = self.disc_laplacian(p)
        diff = self.D*h + self.r*p*(1 - p/self.pmax)
            
        return np.concatenate([x for x in diff])
    
    def solve(self):
        solODE = ode(self.dpdt).set_integrator('dopri5')
        solODE.set_initial_value(self.rho_ex[0],0)
        t = 0
        while t < self.tmax:
            print('t = %.2f'% t,end = '\r')
            t += self.dt
            self.rho.append(self.splitter(solODE.integrate(t)))
        return np.array(self.rho)


# Porous-Fisher model 2 species

class porous_2species:
    
    def __init__(self,Lx,Ly,Nx,Ny,dt,init_rho_1,init_rho_2,tmax,D,r,pmax):
        self.Lx = Lx
        self.Nx = Nx
        self.Ly = Ly
        self.Ny = Ny
        self.dt = dt
        self.dx = Lx/Nx
        self.dy = Ly/Ny
        self.D = D
        self.r = r
        self.pmax = pmax
        self.tmax = tmax
        self.chalf_x = np.linspace(-Lx,Lx,int(2*Nx+1))
        self.cspan_x = np.linspace(-Lx + self.dx/2, Lx - self.dx/2, int(2*Nx))
        self.chalf_y = np.linspace(-Ly,Ly,int(2*Ny+1))
        self.cspan_y = np.linspace(-Ly + self.dy/2, Ly - self.dy/2, int(2*Ny))
        self.X, self.Y = np.meshgrid(self.cspan_x,self.cspan_y)
        self.rho = [np.array([init_rho_1,init_rho_2])]
        self.rho_ex = [np.concatenate([np.concatenate([x for x in init_rho_1]),np.concatenate([x for x in init_rho_2])])]
    
    def splitter(self,rho_expanded):
        return np.array(np.split(rho_expanded,2*self.Nx))
    
    def disc_entropy(self,f,g):
        return f + g
    
    def velocities_x(self,h):
        return -(h[1:,:]-h[:-1,:])/self.dx
    
    def velocities_y(self,h):
        return -(h[:,1:]-h[:,:-1])/self.dy
    
    def flux_x(self,v,g):
        zero_vec = np.zeros(v.shape)
        F = g[:-1,:]*np.maximum(v,zero_vec) + g[1:,:]*np.minimum(v,zero_vec)
        return np.concatenate([[np.zeros(len(F[0]))],F,[np.zeros(len(F[0]))]])
    
    def flux_y(self,v,g):
        zero_vec = np.zeros(v.shape)
        F = g[:,:-1]*np.maximum(v,zero_vec) + g[:,1:]*np.minimum(v,zero_vec)
        return np.transpose(np.concatenate([[np.zeros(len(F))],np.transpose(F),[np.zeros(len(F))]]))
    
    def dpdt(self,t,p_expanded):
        p_ex_1, p_ex_2 = np.split(p_expanded,2)
        p1 = self.splitter(p_ex_1)
        p2 = self.splitter(p_ex_2)
        h = self.disc_entropy(p1,p2)
        Fx_1 = self.flux_x(self.velocities_x(h),p1)
        Fy_1 = self.flux_y(self.velocities_y(h),p1)
        Fx_2 = self.flux_x(self.velocities_x(h),p2)
        Fy_2 = self.flux_y(self.velocities_y(h),p2)
        diff_1 = -self.D*(Fx_1[1:,:]-Fx_1[:-1,:])/self.dx-self.D*(Fy_1[:,1:]-Fy_1[:,:-1])/self.dy + self.r*p1*(1-p1/self.pmax-p2/self.pmax)
        diff_2 = -self.D*(Fx_2[1:,:]-Fx_2[:-1,:])/self.dx-self.D*(Fy_2[:,1:]-Fy_2[:,:-1])/self.dy + self.r*p2*(1-p2/self.pmax-p1/self.pmax)
        return np.concatenate([np.concatenate([x for x in diff_1]),np.concatenate([x for x in diff_2])])   
    
    def solve(self):
        solODE = ode(self.dpdt).set_integrator('dopri5')
        solODE.set_initial_value(self.rho_ex[0],0)
        t = 0
        k = 0
        while t < self.tmax:
            print('t = %.2f'% t,end = '\r')
            t += self.dt
            k += 1
            r1, r2 = np.split(solODE.integrate(t),2)
            self.rho.append(np.array([self.splitter(r1),self.splitter(r2)]))
        return self.rho
    
# Fisher-KPP model 2 species
    
class fisher_2species:

    def __init__(self,Lx,Ly,Nx,Ny,dt,init_rho_1,init_rho_2,tmax,D,r,pmax):
        self.Lx = Lx
        self.Nx = Nx
        self.Ly = Ly
        self.Ny = Ny
        self.dt = dt
        self.dx = Lx/Nx
        self.dy = Ly/Ny
        self.D = D
        self.pmax = pmax
        self.r = r
        self.tmax = tmax
        self.chalf_x = np.linspace(-Lx,Lx,int(2*Nx+1))
        self.cspan_x = np.linspace(-Lx + self.dx/2, Lx - self.dx/2, int(2*Nx))
        self.chalf_y = np.linspace(-Ly,Ly,int(2*Ny+1))
        self.cspan_y = np.linspace(-Ly + self.dy/2, Ly - self.dy/2, int(2*Ny))
        self.X, self.Y = np.meshgrid(self.cspan_x,self.cspan_y)
        self.rho = [np.array([init_rho_1,init_rho_2])]
        self.rho_ex = [np.concatenate([np.concatenate([x for x in init_rho_1]),np.concatenate([x for x in init_rho_2])])]
    
    def splitter(self,rho_expanded):
        return np.array(np.split(rho_expanded,2*self.Nx))
    
    def disc_laplacian(self,f):
        dx, dy = self.dx, self.dy
        lap = np.zeros(f.shape)
        lap[1:-1,1:-1] = (f[2:,1:-1] - 2*f[1:-1,1:-1] + f[:-2,1:-1])/dx**2 + (f[1:-1,2:] - 2*f[1:-1,1:-1] + f[1:-1,:-2])/dy**2
        lap[0,1:-1] = (f[1,1:-1] - f[0,1:-1])/dx**2 + (f[0,2:] - 2*f[0,1:-1] + f[0,:-2])/dy**2
        lap[-1,1:-1] = (-f[-1,1:-1] + f[-2,1:-1])/dx**2 + (f[-1,2:] - 2*f[-1,1:-1] + f[-1,:-2])/dy**2
        lap[1:-1,0] = (f[2:,0] - 2*f[1:-1,0] + f[:-2,0])/dx**2 + (f[1:-1,1] - f[1:-1,0])/dy**2
        lap[1:-1,-1] = (f[2:,-1] - 2*f[1:-1,-1] + f[:-2,-1])/dx**2 +(f[1:-1,-2] - f[1:-1,-1])/dy**2
        lap[0,0] = (f[1,0] - f[0,0])/dx**2 + (f[0,1] - f[0,0])/dy**2
        lap[-1,0] = (f[-2,0] - f[-1,0])/dx**2 + (f[-1,1] - f[-1,0])/dy**2
        lap[-1,-1] = (f[-2,-1] - f[-1,-1])/dx**2 + (f[-1,-2] - f[-1,-1])/dy**2
        lap[0,-1] = (f[1,-1] - f[0,-1])/dx**2 + (f[0,-2] - f[0,-1])/dy**2
        return lap
    
    def dpdt(self,t,p_expanded):
        p_ex_1, p_ex_2 = np.split(p_expanded,2)
        p1 = self.splitter(p_ex_1)
        p2 = self.splitter(p_ex_2)
        h1 = self.disc_laplacian(p1)
        h2 = self.disc_laplacian(p2)
        diff_1 = self.D*h1 + self.r*p1*(1-p1/self.pmax-p2/self.pmax)
        diff_2 = self.D*h2 + self.r*p2*(1-p2/self.pmax-p1/self.pmax)
        return np.concatenate([np.concatenate([x for x in diff_1]),np.concatenate([x for x in diff_2])])
    
    
    def solve(self):
        solODE = ode(self.dpdt).set_integrator('dopri5')
        solODE.set_initial_value(self.rho_ex[0],0)
        t = 0
        k = 0
        while t < self.tmax:
            print('t = %.2f'% t,end = '\r')
            t += self.dt
            k += 1
            r1, r2 = np.split(solODE.integrate(t),2)
            self.rho.append(np.array([self.splitter(r1),self.splitter(r2)]))
        return self.rho