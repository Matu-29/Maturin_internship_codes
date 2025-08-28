###########
# U2H MAP #
###########

import numpy as np


def U2H(u, v, x1d, y1d, waveT, wave_dir_from, s=10, sip=0.01, Hs0=1, Ndir=48, Nf=32, isLHCSsimp=True):

    ### MODULES ###
    
    from scipy import integrate
    import hwffts

    ##############
    # PARAMETERS #
    ##############

    pi = np.pi
    g=9.806 #Gravity acceleration

    theta_p = (90 - (wave_dir_from - 180)) * pi/180
    
    theta = np.linspace(-pi, pi, Ndir) #$\theta$ 
    lowf = 0.04118 #Lowest frequency resolved (in Hz)
    ffactor = 1.1 #controling the grids in frequency
    fp = 1/waveT  #peak frequency (in Hx)
    highf=lowf*(ffactor**32)
    freq=np.linspace(lowf,highf,Nf)
    k=(2*pi*freq)**2/g #Wavenumber $k$ in the unit of m^{-1}. Note 2 pi is needed as f is not angular frequency
    
    #2D grid in $k, \theta$
    k2D,theta2D=np.meshgrid(k,theta)
    k2D=np.transpose(k2D) #Making indexing intuitive, i.e. E2D[i,j] would refer to E2D at k[i] and theta[j]
    theta2D=np.transpose(theta2D)
    
    #Funciton for the distribution in frequency, which is Gaussian. This corresponds to the LHCS model spectrum detailed in Appendix B.
    def gaussian_spec(freq, fp, sip):
        frrel = (freq - fp)/sip
        return np.exp(-1.25*(frrel**2))
    
    #E2D, the 2D wave energy spectrum $\mathcal{E}(k,\theta)$ following the paper's notations 
    #By default, we set it to follow the LHCS model spectrum detailed in Appendix B. Users can change E2D to any other forms on the (k2D,theta2D) grid.
    E2D=gaussian_spec(np.sqrt(g*k2D)/(2*pi), fp, sip)*np.cos((theta2D-theta_p)/2)**(2*s)
    #Normalize so that the corresponding background SWH is Hs0
    Hs0_r=4*np.sqrt(integrate.trapezoid(integrate.trapezoid(E2D*k2D,x=k,axis=0),x=theta)/g)
    E2D=E2D*Hs0**2/Hs0_r**2

    ####################################
    # ZERO PADDING & FOURIER TRANSFORM #
    ####################################
    
    #Parameter for zero padding
    #By default we pad in each direction 100% of the original domain width. A generous padding ensures that the $\hs$ field computed decays to nearly zero at the boundary of the padded domain. 
    #The decay of $\hs$ at the boundary of the padded domain was checked a posteriori in the default example presented here; in extreme cases, users may need to adjust the parameter padportion (decreasing for computational time, or increasing to ensure sufficient decay)/
    padportion=1 #percentage to pad in x or y in each direction in the periphery. 
    
    #Numerical grids after padding
    Nx=len(x1d)
    Ny=len(y1d)
    
    Nxtopad=round(Nx*padportion)
    Nytopad=round(Ny*padportion)
    Nxpad=Nx+2*Nxtopad
    Nypad=Ny+2*Nytopad
    
    dx=x1d[1]-x1d[0]
    xminpad=x1d[0]-dx*(Nxtopad)
    xmaxpad=x1d[-1]+dx*(Nxtopad)
    x1dpad=np.linspace(xminpad,xmaxpad,num=Nxpad)
    
    dy=y1d[1]-y1d[0]
    yminpad=y1d[0]-dy*(Nytopad)
    ymaxpad=y1d[-1]+dy*(Nytopad)
    y1dpad=np.linspace(yminpad,ymaxpad,num=Nypad)
    
    upad=np.zeros((Nxpad,Nypad))
    upad[Nxtopad:Nxtopad+Nx,Nytopad:Nytopad+Ny]=u
    vpad=np.zeros((Nxpad,Nypad))
    vpad[Nxtopad:Nxtopad+Nx,Nytopad:Nytopad+Ny]=v
    
    Nxo=Nx #Nxo is the number of grid points in x before zero padding. 
    Nyo=Ny
    Nxpad=len(x1dpad) #Nxpad is the number of grid points in x after zero padding.
    Nypad=len(y1dpad)
    #Shift x1d and y1d so that they center at 0, for convenience of the application of hwffts. 
    xshift=x1dpad[Nxpad//2-1]
    yshift=y1dpad[Nypad//2-1]
    x1dpad=x1dpad-xshift
    y1dpad=y1dpad-yshift
    
    #Wavenumber for the currents (or Hs), corresponding to the padded domain 
    q1_1d=hwffts.k_of_x(x1dpad) #$q_1$
    q2_1d=hwffts.k_of_x(y1dpad) #$q_2$
    q1_2d,q2_2d=np.meshgrid(q1_1d,q2_1d)
    q1_2d=np.transpose(q1_2d) #Make indexing intuitive, i.e. F[i,j] would mean we want q1 at the ith and q2 at the jth.
    q2_2d=np.transpose(q2_2d)
    
    #polar coordinate of $\bm{q}$, following the paper's notation
    q=np.sqrt(q1_2d**2+q2_2d**2)#wavenumber $q$ of $\bm{q}$
    varphi = np.arctan2(q2_2d, q1_2d) #polar angle $\varphi$ of $\bm{q}$.
    
    #Fourier transform of current velocities. The outputs are evaluated at the grid (q1_2d,q2_2d).
    ut=hwffts.hwfft2(x1dpad, y1dpad, upad)
    vt=hwffts.hwfft2(x1dpad, y1dpad, vpad)


    #######################
    # COMPUTE THE U2H MAP #
    #######################

    #A2D, wave action $\mathcal{A}(k,\theta)$, following the paper's notation 
    A2D=E2D/np.sqrt(g*k2D) #equation (2.4)
    
    #Compute the transfer operator $\bm{L}$ from equation (3.11)
    
    #$\mathcal{P}(\theta)$ from equation (3.8)  
    Pcal=integrate.trapezoid(A2D*k2D**2,x=k,axis=0)
    
    #The cutoff Fourier mode number in the Fourier sum in equation (3.10).  
    #In case the LHCS model is assumed with an integer s (default form of this notebook), $p_n=0$ at $n>s$ (equation B3).
    if isLHCSsimp:
        ncutoff=s 
        
    #In case the LHCS model is not assumed, the parameter ncutoff needs to be input by the user. 
    #ncutoff = $set_your_ncutoff_here$
    

    print("Parameter 'ncutoff' is given as %i." % ncutoff)
    
    if ncutoff>Ndir//2-1:
        warnings.warn(
            "Warning: Parameter'ncutoff' is too large compared to Ndir, and the Fourier sum  in equation (3.10) may be numerically unstable. We suggest increasing Ndir or decreasing ncutoff."
            )
        
    #Get Fourier coefficients pn ($p_n$ in the paper defined in equation (3.10))
    #In this code, we compute pn from a trapezoidal integration. For better computational speed, one can modify so that 
    #the computation of pn can use FFT. 
    pn=np.zeros(2*ncutoff+1)+1j
    nvec=np.arange(-ncutoff,ncutoff+1,1) #Fourier mode number 
    for i in np.arange(len(nvec)):
        ni=nvec[i]
        pn[i]=1/(2*pi)*integrate.trapezoid(Pcal*np.exp(-1j*ni*theta),x=theta)        
        
    #Split $\hat{\mathbf{L}}(\varphi)$ following equation (3.11) in our paper. 
    #The second term in equation (3.11), which is parralel to the wave momentum $\mathbf{P}$:
    P1=np.real(2*pi*pn[nvec==1])
    P2=-np.imag(2*pi*pn[nvec==1])
    LtP1=-32/(g*Hs0**2)*P1
    LtP2=-32/(g*Hs0**2)*P2
    #The first term in equation (3.11), which is parralel to $\mathbf{e_q^{\perp}):
    Lt2=np.zeros([Nxpad,Nypad],dtype=complex)
    for iq1 in np.arange(Nxpad):
        for iq2 in np.arange(Nypad):
            varphii=varphi[iq1,iq2]
            Lt2[iq1,iq2]=16/g/(Hs0**2)*np.sum(nvec*(-1j)**np.abs(nvec)*2*pi*pn*(np.exp(1j*nvec*varphii)))
    Lt21=-Lt2*np.sin(varphi) #The component along $q_1$
    Lt22=Lt2*np.cos(varphi) #The component along $q_2$       
    #Sum the two terms in equation (3.11)
    Lhat1=(LtP1+Lt21) #$\hat{\bm{L}}(\varphi)$ along $q_1$
    Lhat2=(LtP2+Lt22) #$\hat{\bm{L}}(\varphi)$ along $q_2$
    
    #The U2H map
    #Linear multiplication in the $(q_1,q_2)$ space; see equation (1.2)
    hst=Lhat1*ut+Lhat2*vt #In this code, hst is already normalized by $\bar {H}_s$; following the paper's notation, hst is $\hat{h_s}/\bar {H}_s$
    #Inverse Fourier transform
    hs=np.real(hwffts.hwifft2(q1_1d, q2_1d, hst)) #+ Hs0
    
    #So far, the U2H map is conducted on the padded domain, and hs has zero mean over the padded domain. 
    #To be consistent with our formulation, i.e., hs is the spatial anomaly over the unpadded domain, 
    #We subtract from hs its mean over the unpadded domain.
    hs_over_Hs0=(hs-np.mean(hs[Nxtopad:Nxtopad+Nxo,Nytopad:Nytopad+Nyo]))
    
    Hs = hs_over_Hs0 * Hs0 + Hs0


    return np.transpose(Hs[Nxtopad:Nxtopad+Nxo,Nytopad:Nytopad+Nyo]), np.transpose(hs_over_Hs0[Nxtopad:Nxtopad+Nxo,Nytopad:Nytopad+Nyo])