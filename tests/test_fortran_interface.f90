program test_fortran_interface
  use disper26
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_is_finite
  implicit none
  type(physical_parameters), allocatable :: layers(:)
  type(physical_parameters) :: solid,halfspace
  real(c_double), allocatable :: h(:),roots(:)
  real(c_double) :: omega,c,u,hv,ref_c,ref_u,ref_hv,ref_error,error,guess,thickness
  real(c_double) :: max_scaled_error=0.0_c_double
  real(c_double) :: z(5),rho(5),vp(5),vs(5),vph(5),vpv(5),vsh(5),eta(5),z_pad(2,5)
  integer :: scenario,i,status,level,ref_level,count,unit,repeat
  logical :: disc(5)
  character(512) :: message,filename
  character(8) :: wave

  call get_command_argument(1,filename)
  open(newunit=unit,file=trim(filename),status='old',action='read')
  omega=2.0_c_double*acos(-1.0_c_double)/20.0_c_double
  halfspace=create_isotropic_parameters(4.5_c_double,8.0_c_double,3.3_c_double)
  do scenario=1,7
    omega=2.0_c_double*acos(-1.0_c_double)/20.0_c_double
    if (scenario==7) omega=omega*10.0_c_double
    solid=create_isotropic_parameters(3.0_c_double,5.2_c_double,2.5_c_double)
    if (scenario==3 .or. scenario==4) then
      call create_vti_layer_from_depth(2.5_c_double*5.4_c_double**2,2.5_c_double*5.1_c_double**2, &
          0.85_c_double*(2.5_c_double*5.4_c_double**2-2.0_c_double*2.5_c_double*3.0_c_double**2), &
          2.5_c_double*3.0_c_double**2,2.5_c_double*3.2_c_double**2,2.5_c_double, &
          0.0_c_double,10.0_c_double,solid,thickness)
      call close_value(thickness,10.0_c_double)
    end if
    if (allocated(layers)) deallocate(layers,h)
    if (scenario==5) then
      layers=[create_isotropic_parameters(0.0_c_double,1.5_c_double,1.03_c_double),solid]
      h=[3.0_c_double,10.0_c_double]
    else if (scenario==6) then
      allocate(layers(0),h(0))
    else
      layers=[solid]
      h=[10.0_c_double]
    end if
    wave='Rayleigh'
    if (scenario==2 .or. scenario==4) wave='Love'
    do repeat=1,2
      call search_roots(layers,h,halfspace,omega,1.0_c_double,4.49_c_double,500,wave,roots,status,message)
      call success()
    end do
    read(unit,*) count
    if (size(roots)/=count) error stop 'root count mismatch'
    do i=1,count
      read(unit,*) ref_c,ref_u,ref_hv
      call close_value(roots(i),ref_c)
      call compute_group_velocity_ad(layers,h,halfspace,omega,roots(i),wave,u,status,message)
      call success()
      call close_value(u,ref_u)
      call compute_hv_ratio(layers,h,halfspace,omega,roots(i),wave,hv,status,message)
      call success()
      call close_value(hv,ref_hv)
    end do
    print *, 'Layer-cake reference passed:',scenario,count
  end do

  do scenario=1,6
    z=[0.0_c_double,3.0_c_double,3.0_c_double,23.0_c_double,23.0_c_double]
    rho=[2.5_c_double,2.6_c_double,2.7_c_double,2.8_c_double,3.3_c_double]
    vp=[5.2_c_double,5.4_c_double,5.8_c_double,6.2_c_double,8.0_c_double]
    vs=[3.0_c_double,3.1_c_double,3.2_c_double,3.5_c_double,4.5_c_double]
    vph=vp*1.02_c_double
    vpv=vp*0.99_c_double
    vsh=vs*1.03_c_double
    eta=0.9_c_double
    disc=[.false.,.true.,.false.,.true.,.false.]
    guess=4.0_c_double
    if (scenario==5) then
      rho(:2)=1.03_c_double
      vp(:2)=1.5_c_double
      vs(:2)=0
      guess=3.7_c_double
    end if
    wave='Rayleigh'
    if (scenario==2 .or. scenario==4) wave='Love'
    z_pad(1,:)=z
    z_pad(2,:)=-999.0_c_double
    read(unit,*) ref_c,ref_u,ref_hv,ref_level,ref_error
    do repeat=1,2
      ! A noncontiguous depth section exercises Fortran-to-C array handling.
      if (scenario==3 .or. scenario==4) then
        call compute_dispersion_romberg_velocity_vti(z_pad(1,:),rho,vph,vpv,vsh,vs,eta,disc,20.0_c_double, &
            wave,c,u,hv,level,error,status,message,c_guess=guess,phase_only=(repeat==2))
      else
        call compute_dispersion_romberg_velocity_iso(z_pad(1,:),rho,vp,vs,disc,20.0_c_double, &
            wave,c,u,hv,level,error,status,message,c_guess=guess,phase_only=(repeat==2), &
            max_steps=merge(3,5,scenario==6),tolerance=merge(1e-8_c_double,1e-10_c_double,scenario==6), &
            adaptive_root_scan=(scenario/=6),initial_root_window_frac=merge(0.02_c_double,0.0_c_double,scenario==6), &
            extrapolate_start_level=merge(1,0,scenario==6))
      end if
      call success()
      call close_value(c,ref_c)
      call close_value(error,ref_error)
      if (level/=ref_level) error stop 'GRID level mismatch'
      if (repeat==1) then
        call close_value(u,ref_u)
        call close_value(hv,ref_hv)
      else
        if (.not.ieee_is_nan(u) .or. .not.ieee_is_nan(hv)) error stop 'phase_only must return NaNs'
      end if
    end do
    print *, 'GRID reference passed:',scenario
  end do
  close(unit)

  call search_roots(layers,h,halfspace,omega,0.1_c_double,0.2_c_double,50,'Rayleigh',roots,status,message)
  call success()
  if (size(roots)/=0) error stop 'empty interval must return zero roots'
  call search_roots(layers,h,halfspace,omega,1.0_c_double,4.49_c_double,0,'Rayleigh',roots,status,message)
  call invalid()
  call search_roots(layers,h,halfspace,-omega,1.0_c_double,4.49_c_double,50,'Rayleigh',roots,status,message)
  call invalid()
  call search_roots(layers,h,halfspace,omega,1.0_c_double,4.49_c_double,50,'typo',roots,status,message)
  call invalid()
  call search_roots(layers,h,halfspace,omega,4.49_c_double,1.0_c_double,50,'Rayleigh',roots,status,message)
  call invalid()
  call compute_group_velocity_ad(layers,h(:0),halfspace,omega,3.0_c_double,'Rayleigh',u,status,message)
  call invalid()
  if (.not.ieee_is_nan(u)) error stop 'failure output must be NaN'
  call compute_dispersion_romberg_velocity_iso(z,rho(:4),vp,vs,disc,20.0_c_double, &
      'Rayleigh',c,u,hv,level,error,status,message)
  call invalid()
  if (.not.ieee_is_nan(c) .or. level/=-1) error stop 'invalid GRID failure outputs'
  call compute_dispersion_romberg_velocity_iso(z,rho,vp,vs,disc,0.0_c_double, &
      'Rayleigh',c,u,hv,level,error,status,message)
  call invalid()
  disc(1)=.true.
  call compute_dispersion_romberg_velocity_iso(z,rho,vp,vs,disc,20.0_c_double, &
      'Rayleigh',c,u,hv,level,error,status,message)
  call invalid()
  print *, 'Fortran interface checks passed'
  print *, 'Maximum abs(diff)/max(1,abs(reference)):',max_scaled_error
contains
  subroutine success()
    if (status/=DISPER26_OK .or. len_trim(message)/=0) then
      print *, status,trim(message)
      error stop 'unexpected failure'
    end if
  end subroutine
  subroutine invalid()
    if (status/=DISPER26_INVALID_ARGUMENT .or. len_trim(message)==0) then
      print *, status,trim(message)
      error stop 'missing argument error'
    end if
  end subroutine
  subroutine close_value(actual,expected)
    real(c_double), intent(in) :: actual,expected
    if (ieee_is_nan(expected)) then
      if (.not.ieee_is_nan(actual)) error stop 'expected NaN'
    else
      if (.not.ieee_is_finite(actual)) error stop 'nonfinite result'
      max_scaled_error=max(max_scaled_error,abs(actual-expected)/max(1.0_c_double,abs(expected)))
      if (abs(actual-expected)>1e-12_c_double*max(1.0_c_double,abs(expected))) then
        print *, 'actual, expected, difference:',actual,expected,actual-expected
        error stop 'numerical mismatch'
      end if
    end if
  end subroutine
end program
