!========================================================================
! Program: 1D Isentropic Compressible Euler Solver in Finite Background
! using FFT (Pseudo-Spectral Method)
!
! Language      : Fortran 90 with FFTW3
! Author        : Ritwik Mukherjee (ICTS 2025)
! Description   : This code solves the 1D isentropic compressible Euler equations:
!
!     ∂ρ/∂t + ∂/∂x(ρu) = ν∂²ρ/∂x²
!     ∂u/∂t + ∂/∂x(u²/2 + w(ρ)) = ν∂²u/∂x²
!
! where the specific enthalpy w(ρ) = D₀ ρ^δ (isentropic equation of state).
!
! - Pseudo-spectral method is used for spatial derivatives (FFTW)
! - Exponetial and Normal time-stepping (Euler, RK4, IFRK4 options included)
! - 2/3 dealiasing is applied to suppress aliasing error
! - Initial condition: Gaussian density bump with zero velocity
! - Outputs: density and velocity fields at specified intervals
!
!========================================================================

module mod_file
    use, intrinsic :: iso_c_binding 
    include 'fftw3.f03'  ! Include FFTW3 Fortran interface

    !--------------------- Problem Parameters ----------------------
    integer,parameter::N=16384                           ! Number of grid points
    real(kind=8),parameter::nu=1.0d-1                    ! Viscosity (diffusion coefficient)
    real(kind=8),parameter::delta=.4                     ! Equation of state 
    real,parameter::pi = 4.0*atan(1.0)                   ! Pi
    real(kind=8),parameter:: l  = N                      ! Domain length
    real(kind=8),parameter:: D0 = 1.0                    ! Diffusion coefficient
    real(kind=8),parameter:: rhobackground = 1.0         ! Background density
    real(kind=8),parameter:: linspeed = sqrt(delta*D0*rhobackground**delta) ! Characteristic linear speed
    real(kind=8),parameter::tFinal=l/2.2/linspeed        ! Final time of simulation

    !--------------------- Time Discretization ---------------------
    real(kind=8),parameter:: dt = 1.0d-1                ! Time step
    real(kind=8),parameter:: dtt = tFinal/50            ! Time interval for saving output

    !--------------------- Grid & Fourier Space --------------------
    integer,parameter               :: Nh=int(N/2)+1     ! Half grid size (for real-to-complex FFT)
    integer(kind=8),parameter       :: tMax = int(tFinal/dtt)  , navg = int(dtt/dt) ! Time loop counters
    real(kind=8),parameter          :: dx = l/N          ! Grid spacing
    real(kind=8),parameter          :: sigma = 10*dx     ! Width of initial Gaussian bump
    complex(kind=8),parameter       :: zi = (0.0,1.0)     ! Imaginary unit

    !--------------------- Variables -------------------------------
    integer(kind=8)                 :: i , ti , tn 
    real(kind=8),dimension(N)       :: u,x,rho, utemp , rhotemp         ! Real-space fields
    real(kind=8),dimension(2,Nh)    :: kvec, SemiG, SemiGHalf           ! Wavenumbers and exponential filters
    real(kind=8),dimension(Nh)      :: k                                ! 1D wavenumber array
    complex(kind=8),dimension(2,Nh) :: field , rhs                      ! Compact field and RHS in spectral space
    complex(kind=8),dimension(Nh)   :: uk, rhok, dduk , nonlin          ! Spectral fields

    !--------------------- FFTW Plan Handles ------------------------
    integer(kind=8)                 :: plan_forward,plan_backward 
    real(kind=4)                    :: t1,t2                            ! Timing variables
    character(len=100)              :: filename,folder,command         ! Output handling
end module


program burgers
    use mod_file
    call cpu_time(t1) ! Start timing

    open(unit=1,file='para.txt',status='unknown') ! Write parameters
    write(1,*) l,tFinal,dt,dtt
    close(1)

    call initial ! Initialize fields and wavevectors

    !------------------------ FFT Plans ----------------------------
    call dfftw_plan_dft_r2c_1d(plan_forward,N,u,uk,FFTW_ESTIMATE)
    call dfftw_plan_dft_c2r_1d(plan_backward,N,uk,u,FFTW_ESTIMATE+FFTW_PRESERVE_INPUT)

    !------------------------ Initial FFT --------------------------
    call dfftw_execute_dft_r2c(plan_forward,rho,rhok)
    rhok=rhok/real(N)
  
    call dfftw_execute_dft_r2c(plan_forward,u,uk)
    uk = uk/real(N)

    field(1,:) = rhok
    field(2,:) = uk

    !--------------------- Semi-Group Operator --------------------
    SemiG       =  exp(-nu*kvec**2.0*dt)
    SemiGHalf   =  exp(-nu*kvec**2.0*dt/2.0)

    print*,nu ! Print viscosity

    open(unit=10,file='time.txt',status='unknown')
    do ti = 1,tMax       
        ! Convert back to real space
        rhok   = field(1,:)
        uk     = field(2,:)
        call dfftw_execute_dft_c2r(plan_backward,rhok,rho)
        call dfftw_execute_dft_c2r(plan_backward,uk,u)

        ! Output density and velocity
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

            call euler ! Time-step one dt using semi-implicit Euler

        end do 
    end do

    close(10)
    call dfftw_destroy_plan(plan_backward)
    call dfftw_destroy_plan(plan_forward)
    call cpu_time(t2)
    print*,'Time Elapsed=' ,t2-t1
end program


subroutine initial
    use mod_file

    ! Initialize grid and fields
    do i=1,N
        x(i)   = dfloat(i-1-N/2)*dx
        rho(i) = exp(-(x(i)**2/(2.0*sigma**2.0)))   ! Gaussian bump
        u(i)   =  0                                 ! Initially zero velocity
    end do
    rho = rhobackground + rho

    ! Construct wavenumber vector
    do i=1,Nh
        k(i)= 2.0*pi*dfloat(i-1)/l
    end do

    kvec(1,:)=k
    kvec(2,:)=k
end subroutine



subroutine ComputeRhs
    use mod_file
    rhok = field(1,:)
    uk   = field(2,:)

    ! Dealiasing via 2/3 rule
    rhok(int(Nh)-int(Nh/3.0):int(Nh))= 0
    uk(int(Nh)-int(Nh/3.0):int(Nh))  = 0

    ! Back to real space
    call dfftw_execute_dft_c2r(plan_backward,rhok,rho)
    call dfftw_execute_dft_c2r(plan_backward,uk,u)

    ! Compute -d_dx(\rho u)
    call dfftw_execute_dft_r2c(plan_forward,rho*u ,nonlin)
    nonlin = -zi*k*nonlin/dfloat(N) 
    nonlin(int(Nh)-int(Nh/3.0):int(Nh))=0
    rhs(1,:) = nonlin 

    ! Compute -d_dx(w + u²/2)
    call dfftw_execute_dft_r2c(plan_forward, D0*(abs(rho))**(delta) +  u**2.0/2.0  ,nonlin)
    nonlin = -zi*k*nonlin/dfloat(N)    
    nonlin(int(Nh)-int(Nh/3.0):int(Nh))=0
    rhs(2,:) = nonlin 
end subroutine


subroutine euler
    use mod_file
    ! One step of semi-implicit Euler
    call ComputeRhs
    field =  exp(-nu*kvec**2*dt)*(field + rhs*dt) 
end subroutine

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
end subroutine


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
end subroutine
