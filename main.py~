import numpy as np
from scipy.linalg import block_diag
import math

# initial pose
x_0 = np.array([0.,0.,np.deg2rad(0.)]) 

# location of beacon
m_1 = np.array([97.89,70.1])

# covariance of process noise
Q = np.array([
    [  0,  0,  0],     # stdev = .7m
    [  0,  0,  0],
    [  0,  0,  0]])     # stdev = 5 degree

# covariance of measurement noise
R = np.array([
    [0.25,   0.],
    [   0.,   np.deg2rad(1)]])

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
    n = int((_cov.shape[0] - 3)/2) # num of landmarks

    dt = 1
    [vx,vy,vth] = u
    th = _x[2]
    
    _Jf = np.array([
        [1,0,-vx*dt*math.sin(th)-vy*dt*math.cos(th)],
        [0,1,vx*dt*math.cos(th)-vy*dt*math.sin(th)],
        [0,0,1]])

    _Jf = np.column_stack([_Jf,np.zeros((3,n*2))])
    Jf_last_rows = np.column_stack([np.zeros((n*2,3)),np.eye(n*2)])
    Jf = np.vstack([_Jf,Jf_last_rows])

    dx = np.array([vx*dt,vy*dt,vth*dt])
    dx = np.concatenate((dx,np.zeros(n*2)))

    noise = np.random.multivariate_normal(np.zeros(3),Q) 
    noise = np.concatenate((noise,np.zeros(n*2)))

    x = _x + np.dot(Jf,dx) + noise

    x = np.concatenate((x,_x[5:]))

    cov = np.dot(Jf,np.dot(_cov,Jf.T)) + block_diag(Q,np.zeros((n*2,n*2)))

    return [x,cov]

def update(x,cov,data):
    Jg_x = np.array([])
    Jg_m = np.array([])
    error = np.array([])

    for i in data:
        i_mx = i[0]
        i_my = i_mx + 1

        mx = x[i_mx] 
        my = x[i_my] 

        r = math.sqrt((mx-x[0])**2 + (my-x[1])**2)
        th = math.atan2((my-x[1]),(mx-x[0]))-x[2]

        g1_x = (mx-x[0])/r
        g1_y = (my-x[1])/r
        g2_x = (my-x[1])/r**2
        g2_y = (mx-x[0])/r**2

        _Jg_x = np.array([
            [-g1_x,-g1_y,0],
            [g2_x,-g2_y,-1]])
        if Jg_x.size == 0:
            Jg_x = _Jg_x
        else:
            Jg_x = np.vstack([Jg_x,_Jg_x])

        _Jg_m = np.array([
            [g1_x,g1_y],
            [-g2_x,g2_y]])
        Jg_m = block_diag(Jg_m,_Jg_m)

        _error_r = i[1][0] - r
        _error_th = i[1][1] - th
        _error = np.array([_error_r,_error_th])

        if error.size == 0:
            error = _error
        else:
            error = np.vstack([error,_error])

    Jg = np.column_stack([Jg_x,Jg_m])
    
    S = np.dot(Jg,np.dot(cov,Jg.T)) + R
    S_inv = np.linalg.inv(S)
    K = np.dot(cov,np.dot(Jg.T,S_inv))

    x_k = x + np.dot(K,error)
    cov_k = cov - np.dot(K,np.dot(Jg,cov))

    return [x_k,cov_k]

def sensor(x):
    fov = np.deg2rad(180)
    rang = 10

def measure(x,m):
    r = math.sqrt((m[0]-x[0])**2 + (m[1]-x[1])**2)
    th = math.atan2((m[1]-x[1]),(m[0]-x[0])) - x[2]

    z = np.array([r,th])
    return z + np.random.multivariate_normal(np.zeros(2),R)

def new_landmark(x,cov,z):
    # number of landmark in map
    n = int((x.shape[0] - 3)/2)

    # inverse sensor model
    m_x = x[0] + z[0]*math.cos(x[2] + z[1])
    m_y = x[1] + z[0]*math.sin(x[2] + z[1])
    
    dmx_dx = 1.0
    dmx_dy = 0.
    dmx_dth = -z[0]*math.sin(x[2] + z[1])
    dmx_dr = math.cos(x[2] + z[1])
    dmy_dx = 0.
    dmy_dy = 1.0
    dmy_dth = z[0]*math.cos(x[2] + z[1])
    dmy_dr = math.sin(x[2] + z[1])

    new_x = np.concatenate((x,np.array([m_x,m_y])))

    # Jacobian of augmentation function
    _Ja = np.column_stack([np.eye(3+n*2),np.zeros((cov.shape[0],2))])
    Jh_x = np.array([
        [dmx_dx,dmx_dy,dmx_dth],
        [dmy_dx,dmy_dy,dmy_dth]])
    Jh_z = np.array([
        [dmx_dr,dmx_dth],
        [dmy_dr,dmy_dth]])
    
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
        [ 0.49, 0., 0.],     
        [ 0., 0.49, 0.],
        [ 0., 0., np.deg2rad(25)]])     

    u = np.array([0.,0.,np.deg2rad(0.)])

    # initialize new beacon at k=0
    z = measure(x,m_1)
    [x_k,cov_k] = new_landmark(x,cov,z)

    print(0,abs(x_k[0]),math.sqrt(cov_k[0][0]),-math.sqrt(cov_k[0][0]))
#    print(0,abs(x_k[1]),math.sqrt(cov_k[1][1]),-math.sqrt(cov_k[1][1]))
#    print(0,abs(x_k[2]),math.sqrt(cov_k[2][2]),-math.sqrt(cov_k[2][2]))
#    print(0,abs(x_k[3]-97.89),math.sqrt(cov_k[3][3]),-math.sqrt(cov_k[3][3]))
#    print(0,abs(x_k[4]-70.1),math.sqrt(cov_k[4][4]),-math.sqrt(cov_k[4][4]))

    iter = 100
    for num in range(1,iter):
        [x,cov] = [x_k,cov_k]
        u = np.array([0.,0.,np.deg2rad(0.)])

        if u.any() != 0:
            [_x,_cov] = predict(x,u,cov)
        else:
            [_x,_cov] = [x,cov]

        # new measurement and data association
        # data association -> index of landmark
        z = measure(x,m_1) # simulate one detection of landmark
        data = [[3,z]]
        
        if data:
            [x_k,cov_k] = update(_x,_cov,data)
        else:
            [x_k,cov_k] = [_x,_cov]

        print(num,abs(x_k[0]),math.sqrt(cov_k[0][0]),-math.sqrt(cov_k[0][0]))
#        print(num,abs(x_k[1]),math.sqrt(cov_k[1][1]),-math.sqrt(cov_k[1][1]))
#        print(num,abs(x_k[2]),math.sqrt(cov_k[2][2]),-math.sqrt(cov_k[2][2]))
#        print(num,abs(x_k[3]-97.89),math.sqrt(cov_k[3][3]),-math.sqrt(cov_k[3][3]))
#        print(num,abs(x_k[4]-70.1),math.sqrt(cov_k[4][4]),-math.sqrt(cov_k[4][4]))


if __name__ == "__main__":
    main()
