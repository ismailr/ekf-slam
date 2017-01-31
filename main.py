import numpy as np
import math

# initial pose
x_0 = np.array([0,0,0]) 

# location of beacon
m_1 = np.array([97.89,70.1])

# covariance of process noise
Q = np.array([
    [.49,  0,  0],     # stdev = .7m
    [  0,.49,  0],
    [  0,  0, np.deg2rad(25)]])     # stdev = 5 degree

# covariance of measurement noise
R = np.array([
    [0.625,   0],
    [    0,   np.deg2rad(1)]])

def predict(_x,u,_cov):
    """
    process model:
    x_k = f(x_{k-1},u_k)
        = x_{k-1} + u 
        = x_{k-1} + v*dt

    u dalam kerangka acuan robot:
    u = | v_x*dt  | 
        | v_y*dt  |
        | v_th*dt |
    
    u dalam kerangka acuan global:
    u' = Ru

    dengan R adalah matriks rotasi
    R = | cos(th) -sin(th) 0 |
        | sin(th)  cos(th) 0 |
        | 0      0         1 |

    u' = | cos(th) -sin(th) 0 | | v_x*dt  |
         | sin(ht)  cos(th) 0 | | v_y*dt  |
         | 0      0         1 | | v_th*dt |

       = | v_x*cos(th) - v_y*sin(th) |
         | v_x*sin(th) + v_y*cos(th) | * dt
         | v_th                      |
    
    
    x_k = |x_{k-1}|   | v_x*cos(t) - v_y*sin(t)  |
          |y_{k-1}| + | v_x*sin(t) + v_y*cos(t)  | * dt 
          |t_{k-1}|   | v_th                     |

    Linearisasi:
    x_k = |x_{k-1} |   | 1 0   -v_x*dt*sin(theta) - v_y*dt*cos(theta) | | dx| 
          |y_{k-1} | + | 0 1    v_x*dt*cos(theta) - v_y*dt*sin(theta) | | dy|
          |th_{k-1}|   | 0 0                  1                       | |dth|
    """
    dt = 1
    [x,y,th] = _x
    [vx,vy,vth] = u
    
    Jf = np.array([
        [1,0,-vx*dt*math.sin(th)-vy*dt*math.cos(th)],
        [0,1,vx*dt*math.cos(th)-vy*dt*math.sin(th)],
        [0,0,1]])

    dx = np.array([vx*dt,vy*dt,vth*dt])
    
    x_k = _x + np.dot(Jf,dx) + np.random.multivariate_normal(np.zeros(3),Q)
    cov = np.dot(Jf,np.dot(_cov,Jf.T)) + Q

    return [x_k,cov]

def update(x,cov,m):
    if m == None:
        return [x,cov]

    r = math.sqrt((m[0]-x[0])**2 + (m[1]-x[1])**2)
    Jg = np.array([
        [-(m[0]-x[0])/r,-(m[1]-x[1])/r,0,(m[0]-x[0])/r,(m[1]-x[1])/r],
        [(m[1]-x[1])/r**2,-(m[0]-x[0])/r**2,-1,-(m[1]-x[1])/r**2,(m[0]-x[0])/r**2]])
    S = np.dot(Jg,np.dot(cov,Jg.T)) + R
    S_inv = np.linalg.inv(S)
    K = np.dot(cov,np.dot(Jg.T,S_inv))

    error = np.random.multivariate_normal(np.zeros(2),R)
    x_k = x + np.dot(K,error)
    cov_k = cov - np.dot(K,np.dot(Jg,cov))

    return [x_k,cov_k]

def sensor(x):
    fov = np.deg2rad(180)
    rang = 10

def measure(x,m):
    r = math.sqrt((m[0]-x[0])**2 + (m[1]-x[1])**2)
    th = math.atan((m[1]-x[1])/(m[0]-x[0])) - x[2]

    z = np.array([r,th])
    return z + np.random.multivariate_normal(np.zeros(2),R)

def new_landmark(x,cov,z):
    # number of landmark in map
    n = (x.shape[0] - 3)/2

    # inverse sensor model
    m_x = x[0] + z[0]*math.cos(x[2] + z[1])
    m_y = x[1] + z[0]*math.sin(x[2] + z[1])
    
    dmx_dx = 1
    dmx_dy = 0
    dmx_dth = -z[0]*math.sin(x[2] + z[1])
    dmx_dr = math.cos(x[2] + z[1])
    dmy_dx = 0
    dmy_dy = 1
    dmy_dth = z[0]*math.cos(x[2] + z[1])
    dmy_dr = math.sin(x[2] + z[1])

    new_x = np.concatenate((x,np.array([m_x,m_y])))

    # Jacobian of augmentation function
    _Ja = np.column_stack([cov,np.zeros((cov.shape[0],2))])
    Jh_x = np.array([
        [dmx_dx,dmx_dy,dmx_dth],
        [dmy_dx,dmy_dy,dmx_dth]])
    Jh_z = np.array([
        [dmx_dr,dmx_dth],
        [dmy_dr,dmx_dth]])
    
    if n != 0:
        Ja_last_rows = np.column_stack([Jh_x,np.zeros((2,n*2)),Jh_z])
    else:
        Ja_last_rows = np.column_stack([Jh_x,Jh_z])

    Ja = np.vstack([_Ja,Ja_last_rows])

    # joint covariance of process and measurement
    _cov_joint = np.column_stack([cov,np.zeros((cov.shape[0],2))])
    cov_joint_last_rows = np.column_stack([np.zeros((2,cov.shape[1])),R])
    cov_joint = np.vstack([_cov_joint,cov_joint_last_rows])

    # new covariance
    new_cov = np.dot(Ja,np.dot(cov_joint,Ja.T))

    return [new_x,new_cov]
    
def assoc_data():
    None

def main():
    x = x_0
    cov = np.array([
        [  0.5,  0,  0],     
        [  0,  0.5,  0],
        [  0,  0,  np.deg2rad(5)]])     

    u = np.array([10,10,np.deg2rad(10)])

    # get new measurement
    # do association data
    
    # suppose there's new beacon
    z = measure(x,m_1)

    if u.any() != 0:
        [_x,_cov] = predict(x,u,cov)
    else:
        [_x,_cov] = [x,cov]

    [x_k,cov_k] = update(_x,_cov,None)
    [x_k_new,cov_k_new] = new_landmark(x_k,cov_k,z)

    x_k_new_print = x_k_new
    x_k_new_print[2] = np.rad2deg(x_k_new_print[2])
    print(x_k_new_print)
    print(cov_k_new)

if __name__ == "__main__":
    main()
