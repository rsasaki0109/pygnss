import math
import numpy as np
import pandas as pd
D2R = np.pi/180.0
R2D = 180.0/np.pi
#outputs:
#    pos(type:ndarray)
def  read_rtklibpos(path):
    df=pd.read_table(path,header=None,sep='\s+',comment='%')
    
    hms=df.iloc[:,1].str.split(":")
    t=[float(hms.iloc[i][0])*60**2+float(hms.iloc[i][1])*60+float(hms.iloc[i][2]) for i, unused_list in enumerate(hms)]

    df2=pd.concat([df.iloc[:,0].str.replace("/", ""),pd.Series(t), df.iloc[:,2:]], axis=1)

    pos=df2.values.astype("float64")
    return pos


def xyz2llh(xyz):

    a = 6378137.0000    # earth radius in meters    (WGS84)
    b = 6356752.3142    # earth semiminor in meters    (WGS84)
    
    x2 = xyz[:,0]**2
    y2 = xyz[:,1]**2
    z2 = xyz[:,2]**2
    
    e   = np.sqrt(1-(b/a)**2)
    b2  = b*b
    e2  = e**2
    ep  = e*(a/b)
    r   = np.sqrt(x2+y2)
    r2  = r*r
    E2  = a**2-b**2
    F   = 54*b2*z2
    G   = r2+(1-e2)*z2-e2*E2
    c   = ((e2*e2)*F*r2)/(G**3)
    s   = (1+c+np.sqrt(c*c+2*c))**(1/3)
    P   = F/(3*(s+1/s+1)**2*G*G)
    Q   = np.sqrt(1+2*e2*e2*P)
    ro  = -(P*e2*r)/(1+Q)+np.sqrt((a*a/2)*(1+1/Q)-(P*(1-e2)*z2)/(Q*(1+Q))-P*r2/2)
    tmp = (r-e2*ro)**2
    U   = np.sqrt(tmp+z2)
    V   = np.sqrt(tmp+(1-e2)*z2)
    zo  = (b2*xyz[:,2])/(a*V)    
    
    h   = U*(1-b2/(a*V))                              # ellipsoidal height
    lat = np.arctan((xyz[:,2]+ep*ep*zo)/r)*R2D        # latitude [degree]
    
    # longitude [degree]
    lon      = np.arctan(xyz[:,1]/xyz[:,0])*R2D           
    ind      = np.logical_and(xyz[:,0]<0,xyz[:,1]>=0)
    lon[ind] = lon[ind]+180
    ind      = np.logical_and(xyz[:,0]<0,xyz[:,1]<0)
    lon[ind] = lon[ind]-180    
    
    h   = h.reshape((1,-1))
    lat = lat.reshape((1,-1))
    lon = lon.reshape((1,-1))
    llh = np.hstack((lat.T,lon.T,h.T))
    
    return llh


#inputs:
#    xyz(n x 3)
def xyz2enu(xyz,orgxyz):
       n = xyz.shape[0]
       difxyz = xyz-np.tile(orgxyz,(n,1))
       orgllh = xyz2llh(orgxyz)
       phi = orgllh[0,0]*D2R
       lam = orgllh[0,1]*D2R  
       sinphi = np.sin(phi)
       cosphi = np.cos(phi)
       sinlam = np.sin(lam)
       coslam = np.cos(lam)   
       R = [[-sinlam        , coslam         ,0     ],
            [-sinphi*coslam ,-sinphi*sinlam  ,cosphi],
            [ cosphi*coslam , cosphi*sinlam  ,sinphi]]
       enu = np.dot(R,difxyz.T).T
       return enu   

def llh2xyz(llh):
    
    if len(llh.shape) == 1: # if input is orgllh
        llh = llh.reshape((1,-1))
    
    a = 6378137.0000        # earth radius in meters    (WGS84)
    b = 6356752.3142        # earth semiminor in meters    (WGS84)
    e = np.sqrt(1-(b/a)**2) 
     
    sinphi = np.sin(llh[:,0]*D2R)
    cosphi = np.cos(llh[:,0]*D2R)
    coslam = np.cos(llh[:,1]*D2R)
    sinlam = np.sin(llh[:,1]*D2R)    
    tan2phi = (np.tan(llh[:,0]*D2R))**2
    tmp = 1-e**2
    tmpden = np.sqrt(1+tmp*tan2phi)        
    x = (a*coslam)/tmpden+llh[:,2]*coslam*cosphi
    y = (a*sinlam)/tmpden+llh[:,2]*sinlam*cosphi
    z = (a*tmp*sinphi)/np.sqrt(1-e*e*sinphi*sinphi)+llh[:,2]*sinphi  
         
    x = x.reshape((1,-1))
    y = y.reshape((1,-1))
    z = z.reshape((1,-1))
    xyz = np.hstack((x.T,y.T,z.T))
    
    return xyz

def llh2enu(llh,orgllh):
    xyz = llh2xyz(llh)
    orgxyz = llh2xyz(orgllh)
    enu = xyz2enu(xyz,orgxyz)
    return enu
