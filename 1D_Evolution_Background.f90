module mod_file
    use, intrinsic :: iso_c_binding 
    include 'fftw3.f03'
	integer,parameter::N=16384
	real(kind=8),parameter::nu=1.0d-1
	real(kind=8),parameter::delta=.40000000000000000000
    real,parameter::pi = 4.0*atan(1.0)  
    real(kind=8),parameter:: l  = N  
    real(kind=8),parameter:: D0 = 1.0
    real(kind=8),parameter:: rhobackground = 1.0
    real(kind=8),parameter:: linspeed = sqrt(delta*D0*rhobackground**delta)
	real(kind=8),parameter::tFinal=l/2.2/linspeed

    real(kind=8),parameter:: dt = 1.0d-1, dtt = tFinal/50
    !=================================================================
    integer,parameter               ::      Nh=int(N/2)+1
    integer(kind=8),parameter       ::      tMax = int(tFinal/dtt)  , navg = int(dtt/dt)
    real(kind=8),parameter          ::      dx = l/N 
    real(kind=8),parameter          ::      sigma = 10*dx
    complex(kind=8),parameter       ::      zi = (0.0,1.0) 
    integer(kind=8)                 ::      i , ti , tn 
    real(kind=8),dimension(N)       ::      u,x,rho, utemp , rhotemp
    real(kind=8),dimension(2,Nh)    ::      kvec, SemiG, SemiGHalf
    real(kind=8),dimension(Nh)      ::      k
    complex(kind=8),dimension(2,Nh) ::      field , rhs
    complex(kind=8),dimension(Nh)   ::      uk, rhok, dduk , nonlin
    integer(kind=8)                 ::      plan_forward,plan_backward 
    real(kind=4)                    ::      t1,t2
    character(len=100)              ::      filename,folder,command
end module
program burgers
    use mod_file
    call cpu_time(t1)

    open(unit=1,file='para.txt',status='unknown')
    write(1,*) l,tFinal,dt,dtt
    close(1)

    call initial 

    call dfftw_plan_dft_r2c_1d(plan_forward,N,u,uk,FFTW_ESTIMATE)
    call dfftw_plan_dft_c2r_1d(plan_backward,N,uk,u,FFTW_ESTIMATE+FFTW_PRESERVE_INPUT)

    
    call dfftw_execute_dft_r2c(plan_forward,rho,rhok)
    rhok=rhok/real(N)
  
    call dfftw_execute_dft_r2c(plan_forward,u,uk)
    uk = uk/real(N)

  



    field(1,:) = rhok
    field(2,:) = uk

    SemiG       =  exp(-nu*kvec**2.0*dt)
    SemiGHalf   =  exp(-nu*kvec**2.0*dt/2.0)

    !===================================================================
    print*,nu
  

    open(unit=10,file='time.txt',status='unknown')
    do ti = 1,tMax       

        !!---------------------------------------
        !! Opening and Writing files
        !!---------------------------------------
        rhok   = field(1,:)
        uk     = field(2,:)
        call dfftw_execute_dft_c2r(plan_backward,rhok,rho)
        call dfftw_execute_dft_c2r(plan_backward,uk,u)
        write(filename, '(A,I0,A)') 'fieldrho', ti , '.txt'
        open(unit=51,file=filename,status='unknown')
        write(51,*)rho
        close(51)
        write(filename, '(A,I0,A)') 'fieldu', ti , '.txt'
        open(unit=51,file=filename,status='unknown')
        write(51,*)u
        close(51)

        open(unit=1,file='time.txt',status='unknown')
        write(1,*) (ti-1)*dtt

        write(*,'(A,F5.2,A,F6.2)')' Time    =   ',(ti-1)*dtt,'  Mass  =',sum(rho)*dx
        call cpu_time(t2)
        print*,'Time taken = ', (t2-t1) , 'seconds'

        if (isnan(maxval(u))) then 
            print *, "NaN detected"
            exit
        end if
        
        do tn = 1, navg 
             if(mod(tn,100)==0) then
                call cpu_time(t2)
                print*,'Time=',ti*dtt+tn*dt, 'Out of =', tFinal, 'Time taken = ', (t2-t1) , 'seconds'
                if (isnan(maxval(u))) then 
                        print *, "NaN detected"
                        exit
                end if
              end if
            !--------------------------------------------------------------------
            call euler
            !-------------------------------------------------------------------   
        end do 

    end do
    close(10)
    call dfftw_destroy_plan(plan_backward)
    call dfftw_destroy_plan(plan_forward)
    call cpu_time(t2)
    print*,'Time Elapsed=' ,t2-t1
end program burgers

subroutine initial
    use mod_file

    do i=1,N
        x(i)   = dfloat(i-1-N/2)*dx
        rho(i) = exp(-(x(i)**2/(2.0*sigma**2.0)))
        u(i)   =  0*rho(i)
    end do
    rho = rhobackground + rho!/sum(rho*dx)

    do i=1,Nh
        k(i)= 2.0*pi*dfloat(i-1)/l
    end do

    kvec(1,:)=k
    kvec(2,:)=k


end subroutine initial

subroutine ComputeRhs
    use mod_file
    rhok = field(1,:)
    uk   = field(2,:)
    !--------------- dealiasing----------------------------------
    rhok(int(Nh)-int(Nh/3.0):int(Nh))= 0
    uk(int(Nh)-int(Nh/3.0):int(Nh))  = 0
    !-----------------------------------------------------------
    ! Convert to real fields 
    
    call dfftw_execute_dft_c2r(plan_backward,rhok,rho)
    call dfftw_execute_dft_c2r(plan_backward,uk,u)

    !!-----------------------Nolinear Term----------------------------
    ! 1st Equation 
    call dfftw_execute_dft_r2c(plan_forward,rho*u ,nonlin)
    nonlin = -zi*k*nonlin/dfloat(N) !  
    nonlin(int(Nh)-int(Nh/3.0):int(Nh))=0
    rhs(1,:) = nonlin 

    ! 2nd equation 
    call dfftw_execute_dft_r2c(plan_forward, D0*(abs(rho))**(delta) +  u**2.0/2.0  ,nonlin)
    nonlin = -zi*k*nonlin/dfloat(N)    
    nonlin(int(Nh)-int(Nh/3.0):int(Nh))=0
    rhs(2,:) = nonlin 
    


    !-------------------------------------------------------------------



    !!-------------------------------------------------------------------------
end subroutine ComputeRhs
subroutine euler
    use mod_file
    !--------------------------------------------------------------------------
    call ComputeRhs
    field =  exp(-nu*kvec**2*dt)*(field + rhs*dt) 
end subroutine euler


subroutine RK4
    use mod_file
    complex(kind=8),dimension(2,Nh):: k1,k2,k3,k4,field_old
        field_old = field
        call ComputeRhs
        k1 = rhs - nu*field*kvec**2.0 
        field = field_old + 0.5d0*dt*k1
        call ComputeRhs
        k2 = rhs - nu*field*kvec**2.0 
        field = field_old + 0.5d0*dt*k2
        call ComputeRhs
        k3 = rhs - nu*field*kvec**2.0 
        field = field_old + dt*k3
        call ComputeRhs
        k4 = rhs - nu*field*kvec**2.0 

        field = field_old + dt*(1d0/6.0d0)*(k1+2.0*k2+2.0*k3+k4)

end subroutine RK4


subroutine IFRK4
    use mod_file
    complex(kind=8),dimension(2,Nh):: k1,k2,k3,k4,field_old
        field_old = field
        call ComputeRhs
        k1 = rhs 
        field = SemiGHalf*(field_old + 0.5d0*dt*k1)
        call ComputeRhs
        k2 = rhs
        field = SemiGHalf*field_old + 0.5d0*dt*k2
        call ComputeRhs
        k3 = rhs 
        field = SemiG*field_old + SemiGHalf*dt*k3
        call ComputeRhs
        k4 = rhs 

        field = SemiG*field_old + dt*(1d0/6.0d0)*(SemiG*k1+2.0*SemiGHalf*(k2 + k3)+k4)

end subroutine IFRK4
