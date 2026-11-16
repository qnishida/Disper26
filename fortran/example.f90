program example
  use disper26
  implicit none
  type(physical_parameters) :: layers(1), halfspace
  real(c_double), allocatable :: roots(:)
  real(c_double) :: omega,u,hv,c,error,period
  real(c_double) :: depths(3),rhos(3),vps(3),vss(3)
  integer :: status,level,i
  character(512) :: message

  layers(1)=create_isotropic_parameters(3.0_c_double,5.2_c_double,2.5_c_double)
  halfspace=create_isotropic_parameters(4.5_c_double,8.0_c_double,3.3_c_double)
  omega=2.0_c_double*acos(-1.0_c_double)/20.0_c_double
  call search_roots(layers,[10.0_c_double],halfspace,omega,2.0_c_double,4.49_c_double, &
      500,'Rayleigh',roots,status,message)
  call check_status()
  if (size(roots)==0) error stop 'No roots in the requested interval'
  c=roots(1)
  call compute_group_velocity_ad(layers,[10.0_c_double],halfspace,omega,c,'Rayleigh',u,status,message)
  call check_status()
  call compute_hv_ratio(layers,[10.0_c_double],halfspace,omega,c,'Rayleigh',hv,status,message)
  call check_status()
  print *, 'Layer-cake c, U [km/s], H/V:',c,u,hv

  depths=[0.0_c_double,10.0_c_double,30.0_c_double]
  rhos=[2.5_c_double,2.8_c_double,3.3_c_double]
  vps=[5.2_c_double,6.2_c_double,8.0_c_double]
  vss=[3.0_c_double,3.5_c_double,4.5_c_double]
  do i=1,3
    period=10.0_c_double+5.0_c_double*i
    call compute_dispersion_romberg_velocity_iso(depths,rhos,vps,vss,[.false.,.false.,.false.], &
        period,'Rayleigh',c,u,hv,level,error,status,message,c_guess=4.0_c_double)
    call check_status()
    print *, 'GRID period [s], c, U [km/s], H/V, level, relative error:',period,c,u,hv,level,error
  end do
contains
  subroutine check_status()
    if (status/=DISPER26_OK) then
      print *, trim(message)
      error stop 1
    end if
  end subroutine
end program
