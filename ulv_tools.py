

""" Python tools for ULV Matrix Decompositions based on 

UTV Tools
Matlab Templates for
Rank-Revealing UTV Decompositions
from Per Christian Hansen

http://www2.compute.dtu.dk/~pcha/
http://people.compute.dtu.dk/pcha/software.html

"""


import numpy as np

def pythag(y,z):
    rmax = np.max(np.abs(np.concatenate((y,z),axis=None))) 
    if (rmax == 0):
        x = 0
    else:
        x = rmax*np.sqrt((y/rmax)**2 + (z/rmax)**2)

    return x


def app_giv(v1,v2,c,s):
    u1 = c*v1 + s*v2
    u2 = c*v2 - s*v1
    return u1,u2 


def gen_giv(a,b):

    if (a == 0):
        c = 0
        s = 1
        r = b
    else:
        if (np.abs(a) > np.abs(b)):
            t  = b/a
            tt = np.sqrt(1 + t*t)
            c  = 1/tt
            s  = c*t
            r  = a*tt
        else:
            t  = a/b
            tt = np.sqrt(1 + t*t)
            s  = 1/tt
            c  = s*t
            r  = b*tt

    return c,s,r

def ulv_rdef(L,V,U,r,umin):

    # Initialize.
    [n,n] = L.shape    

    for i in range(1,r,1):
        # Transform umin to e_n.
        c,s,umin[i] = gen_giv(umin[i],umin[i-1])

        # Apply rotation to L on the left.
        L[i-1,:i+1],L[i,:i+1] = app_giv(L[i-1,:i+1],L[i,:i+1],c,-s)

        # Apply rotation to U on the right.
        U[:,i-1],U[:,i] = app_giv(U[:,i-1],U[:,i],c,-s)

        # Restore L to lower triangular form using rotation on the right.
        c,s,L[i-1,i-1] = gen_giv(L[i-1,i-1],L[i-1,i])
        L[i-1,i] = 0                             # Eliminate L(i-1,i).
        L[i:n,i-1],L[i:n,i] = app_giv(L[i:n,i-1],L[i:n,i],c,s)

        # Apply rotation to V on the right.
        temp1,temp2 = app_giv(V[:n,i-1],V[:n,i],c,s)
        V[:n,i-1],V[:n,i] = app_giv(V[:n,i-1],V[:n,i],c,s)

    return L,V,U



def ccvl(R):

    # Initialize.
    n,n = R.shape
    if (n==0):
        smin = []
        vmin = []
        return
    
    eps = np.finfo(float).eps
    #eps = 2.2204e-16

    for i in range(n):
        if R[i,i]==0:
            R[i,i]=eps

    v = np.zeros((n,))

    # First element is treated special.
    v[0] = 1/R[0,0]
    vnorm = np.abs(v[0])
    p     = v[0]*R[0,:]  


    # Process rows of matrix one by one.
    for i in range(1,n-1,1):
    
        u     = R[i,i+1:]
        utp   = np.dot(u,p[i+1:])
        gamma = R[i,i]
        xi    = p[i]
        phi   = 1 + np.linalg.norm(u)**2
        pnorm = np.linalg.norm(p[i+1:])
        alpha = xi*phi - gamma*utp

        if (alpha == 0):
            beta = gamma**2 *(vnorm**2 + pnorm**2) - (xi**2+1)*phi
            if (beta > 0):
                s = 1
                c = 0
            else:
                s = 0
                c = 1
        else:
            beta = gamma**2 *(vnorm**2 + pnorm**2) + (xi**2-1)*phi - 2*xi*gamma*utp
            eta  = beta/pythag(beta,2*alpha)      # eta is cos(2a).
            s    = -np.sign(alpha)*np.sqrt((1+eta/2))
            c    = np.sqrt((1-eta)/2)


        v[:i+1]   = np.concatenate((s*v[:i],(c-s*xi)/gamma),axis=None)
        vnorm    = pythag(s*vnorm,v[i])
        p[i+1:n] = s*p[i+1:n] + v[i]*u

    # Last step is the same as in incremental condition estimation.
    alpha = p[n-1]
    gamma = R[n-1,n-1]
    if (alpha == 0):
        beta = gamma**2 *vnorm**2 - 1
        if (beta > 0):
            s = 1
            c = 0
            lambda_max = beta + 1
        else:
            s = 0
            c = 1
            lambda_max = 1
    else:
        beta = gamma**2*vnorm**2 + alpha**2 - 1
        eta  = beta/pythag(beta,2*alpha)
        lambda_max = 0.5*(beta + pythag(beta,2*alpha)) + 1
        s = -np.sign(alpha)*np.sqrt((1+eta)/2)
        c = np.sqrt((1-eta)/2)

    v     = np.concatenate((s*v[:n-1],(c-s*alpha)/gamma),axis=None)
    vnorm = np.sqrt(lambda_max)/np.abs(gamma)

    # Compute and normalize right nullvector.
    vmin = np.linalg.solve(R,(v/vnorm))
    smin = 1/np.linalg.norm(vmin)
    vmin = smin*vmin

    return smin,vmin


def  ulv_ref(L,V,U,r):

    #  Initialize.
    n,n = L.shape
    # Flip last row of L up.
    for i in range(r-2,-1,-1):
        # Apply rotation to L on the left.
        c,s,L[i,i] = gen_giv(L[i,i],L[r-1,i])
        L[r-1,i] = 0;                               # Eliminate L(r-1,i).
        L[i,:i],L[r-1,:i] = app_giv(L[i,:i],L[r-1,:i],c,s)
        L[i,r-1],L[r-1,r-1]         = app_giv(L[i,r-1],L[r-1,r-1],c,s)

        # Apply rotation to U on the right.
        U[:,i],U[:,r-1] = app_giv(U[:,i],U[:,r-1],c,s)

        
    # Flip last column of L down.
    for i in range(r-1):
        # Restore L to lower triangular form using rotation on the right.
        c,s,L[i,i] = gen_giv(L[i,i],L[i,r-1])
        L[i,r-1] = 0                               # Eliminate L(i,r-1).
        L[i+1:,i],L[i+1:,r-1] = app_giv(L[i+1:,i],L[i+1:,r-1],c,s)

        # % Apply rotation to V on the right.
        V[:,i],V[:,r-1] = app_giv(V[:,i],V[:,r-1],c,s)

    return L,V,U


def hulv(A,tol_rank=None,tol_ref=1e-04,max_ref=0,fixed_rank=0):
    """ 
      hulv --> Stewart's high-rank-revealing ULV algorithm.
    
      <Synopsis>
        [p,L,V,U] = hulv(A)
        [p,L,V,U] = hulv(A,tol_rank)
        [p,L,V,U] = hulv(A,tol_rank,tol_ref,max_ref)
        [p,L,V,U] = hulv(A,tol_rank,tol_ref,max_ref,fixed_rank)
    
      <Description>
        Computes a rank-revealing ULV decomposition of an m-by-n matrix A
        with m >= n, where the algorithm is optimized for numerical rank p
        close to n. In the two-sided orthogonal decomposition, the n-by-n
        matrix L is lower triangular and will reveal the numerical rank p
        of A. Thus, the norm of the (2,1) and (2,2) blocks of L are of the
        order sigma_(p+1). U and V are unitary matrices, where only the
        first n columns of U are computed.
    
      <Input Parameters>
        1. A          --> m-by-n matrix (m >= n);
        2. tol_rank   --> rank decision tolerance;
        3. tol_ref    --> upper bound on the 2-norm of the off-diagonal block
                          L(p+1:n,1:p) relative to the Frobenius-norm of L;
        4. max_ref    --> max. number of refinement steps per singular value
                          to achieve the upper bound tol_ref;
        5. fixed_rank --> deflate to the fixed rank given by fixed_rank instead
                          of using the rank decision tolerance;
    
        Defaults: tol_rank = sqrt(n)*norm(A,1)*eps;
                  tol_ref  = 1e-04;
                  max_ref  = 0;
    
      <Output Parameters>
        1.   p       --> numerical rank of A;
        2-4. L, V, U --> the ULV factors such that A = U*L*V';

    
      <Algorithm>
        The rectangular matrix A is preprocessed by a QL factorization, A = U*L.
        Then deflation and refinement (optional) are employed to produce a
        rank-revealing decomposition. The deflation procedure is based on the
        generalized LINPACK condition estimator, and the refinement steps on
        QR-iterations.
    
      <See Also>
        hulv_a --> An alternative high-rank-revealing ULV algorithm.

      <References>
      [1] G.W. Stewart, "Updating a Rank-Revealing ULV Decomposition",
          SIAM J. Matrix Anal. and Appl., 14 (1993), pp. 494--499.
    
      <Revision>
        Ricardo D. Fierro, California State University San Marcos
        Per Christian Hansen, IMM, Technical University of Denmark
        Peter S.K. Hansen, IMM, Technical University of Denmark
    
        Last revised: June 22, 1999


    """


    m,n = A.shape
    if m*n==0:
        raise Exception("Empty input matrix A not allowed.")
    elif (m<n):
        raise Exception('The system is underdetermined; use HURV on the transpose of A.')
    
    if tol_rank is None:
        eps = np.finfo(float).eps
        tol_rank = np.sqrt(n)*np.linalg.norm(A,ord=1)*eps

    if (tol_rank != abs(tol_rank)) | (tol_ref != abs(tol_ref)):
        raise Exception('Requires positive values for tol_rank and tol_ref.')

    if (max_ref != abs(round(max_ref))):
        raise Exception('Requires positive integer value for max_ref.')
    

    U,R = np.linalg.qr(A[:m,n::-1])
    U = U[:m,n::-1]
    L = R[n::-1,n::-1]
    V = np.eye(n)



    # Rank-revealing procedure.

    # Initialize.
    smin_p_plus_1 = 0                             # No (n+1)th singular value.
    norm_tol_ref  = np.linalg.norm(L,'fro')*tol_ref/np.sqrt(n) # Value used to verify ...

    # Estimate of the n'th singular value and the corresponding left ...
    # singular vector via the generalized LINPACK condition estimator.
    smin,umin = ccvl(np.transpose(L[:n,:n]))

    p = n                               # Init. loop to full rank n.  

    while ((smin < tol_rank) &(p > fixed_rank)):
        # Apply deflation procedure to p'th row of L in the ULV decomposition.
        L,V,U = ulv_rdef(L,V,U,p,umin)

        # Refinement loop.
        num_ref = 0                                 # Init. refinement counter.
        while (np.linalg.norm(L[p-1,:p-1]) > norm_tol_ref) & (num_ref < max_ref):
            print('yes')
            # Apply one QR-iteration to p'th row of L in the ULV decomposition.
            L,V,U = ulv_ref(L,V,U,p)
            num_ref = num_ref + 1

        # New rank estimate after the problem has been deflated.
        p = p - 1
        smin_p_plus_1 = smin

        # Estimate of the p'th singular value and the corresponding left ...
        # singular vector via the generalized LINPACK condition estimator.
        if (p > 0):
            smin,umin = ccvl(np.transpose(L[:p,:p]))
        else:
            smin = 0       # No 0th singular value.


    return p,L,V,U
