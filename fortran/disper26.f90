! Basic flat-model interface. See README.md for units and error semantics.
module disper26
  use, intrinsic :: iso_c_binding, only: c_double, c_int, c_char, c_null_char
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  implicit none
  private
  public :: c_double, physical_parameters, create_isotropic_parameters, create_vti_layer_from_depth
  public :: search_roots, compute_group_velocity_ad, compute_hv_ratio
  public :: compute_dispersion_romberg_velocity_iso, compute_dispersion_romberg_velocity_vti
  integer, parameter, public :: DISPER26_OK=0, DISPER26_INVALID_ARGUMENT=1
  integer, parameter, public :: DISPER26_BUFFER_TOO_SMALL=2, DISPER26_CALCULATION_FAILED=3
  integer(c_int), parameter :: message_capacity=512

  type :: physical_parameters
    real(c_double) :: rho=0, c11=0, c13=0, c33=0, c44=0, c66=0
  end type

  interface
    function c_roots(n,p,h,hs,omega,c_min,c_max,n_search,wave,roots,capacity,count,msg,msgcap) &
        bind(C,name="disper26_search_roots") result(status)
      import c_double, c_int, c_char
      integer(c_int), value :: n,n_search,wave,capacity,msgcap
      real(c_double), intent(in) :: p(*),h(*),hs(*)
      real(c_double), value :: omega,c_min,c_max
      real(c_double), intent(out) :: roots(*)
      integer(c_int), intent(out) :: count
      character(kind=c_char), intent(out) :: msg(*)
      integer(c_int) :: status
    end function
    function c_group_velocity_ad(n,p,h,hs,omega,cp,wave,value,msg,msgcap) &
        bind(C,name="disper26_compute_group_velocity_ad") result(status)
      import c_double, c_int, c_char
      integer(c_int), value :: n,wave,msgcap
      real(c_double), intent(in) :: p(*),h(*),hs(*)
      real(c_double), value :: omega,cp
      real(c_double), intent(out) :: value
      character(kind=c_char), intent(out) :: msg(*)
      integer(c_int) :: status
    end function
    function c_hv_ratio(n,p,h,hs,omega,cp,wave,value,msg,msgcap) &
        bind(C,name="disper26_compute_hv_ratio") result(status)
      import c_double, c_int, c_char
      integer(c_int), value :: n,wave,msgcap
      real(c_double), intent(in) :: p(*),h(*),hs(*)
      real(c_double), value :: omega,cp
      real(c_double), intent(out) :: value
      character(kind=c_char), intent(out) :: msg(*)
      integer(c_int) :: status
    end function
    function c_grid_iso(n,z,p,flags,period,wave,guess,steps,tol,phase_only,adaptive,window,start, &
        c,u,hv,level,error,msg,msgcap) bind(C,name="disper26_compute_dispersion_romberg_velocity_iso") result(status)
      import c_double, c_int, c_char
      integer(c_int), value :: n,wave,steps,phase_only,adaptive,start,msgcap
      integer(c_int), intent(in) :: flags(*)
      real(c_double), intent(in) :: z(*),p(*)
      real(c_double), value :: period,guess,tol,window
      real(c_double), intent(out) :: c,u,hv,error
      integer(c_int), intent(out) :: level
      character(kind=c_char), intent(out) :: msg(*)
      integer(c_int) :: status
    end function
    function c_grid_vti(n,z,p,flags,period,wave,guess,steps,tol,phase_only,adaptive,window,start, &
        c,u,hv,level,error,msg,msgcap) bind(C,name="disper26_compute_dispersion_romberg_velocity_vti") result(status)
      import c_double, c_int, c_char
      integer(c_int), value :: n,wave,steps,phase_only,adaptive,start,msgcap
      integer(c_int), intent(in) :: flags(*)
      real(c_double), intent(in) :: z(*),p(*)
      real(c_double), value :: period,guess,tol,window
      real(c_double), intent(out) :: c,u,hv,error
      integer(c_int), intent(out) :: level
      character(kind=c_char), intent(out) :: msg(*)
      integer(c_int) :: status
    end function
  end interface
contains
  pure function create_isotropic_parameters(vs,vp,rho) result(p)
    real(c_double), intent(in) :: vs,vp,rho
    type(physical_parameters) :: p
    p%rho=rho
    p%c44=rho*vs*vs
    p%c66=p%c44
    p%c11=rho*vp*vp
    p%c33=p%c11
    p%c13=p%c11-2.0_c_double*p%c44
  end function

  pure subroutine create_vti_layer_from_depth(a,c,f,l,n,rho,top_depth,bottom_depth,layer,thickness)
    real(c_double), intent(in) :: a,c,f,l,n,rho,top_depth,bottom_depth
    type(physical_parameters), intent(out) :: layer
    real(c_double), intent(out) :: thickness
    layer=physical_parameters(rho,a,f,c,l,n)
    thickness=bottom_depth-top_depth
  end subroutine

  pure function pack_parameter(p) result(values)
    type(physical_parameters), intent(in) :: p
    real(c_double) :: values(6)
    values=[p%rho,p%c11,p%c13,p%c33,p%c44,p%c66]
  end function

  function wave_code(wave_type) result(wave)
    character(*), intent(in) :: wave_type
    integer(c_int) :: wave
    select case(trim(wave_type))
    case('Rayleigh','rayleigh')
      wave=0
    case('Love','love')
      wave=1
    case default
      wave=-1
    end select
  end function

  subroutine copy_message(buffer,message)
    character(kind=c_char), intent(in) :: buffer(:)
    character(*), intent(out) :: message
    integer :: i
    message=''
    do i=1,min(size(buffer),len(message))
      if (buffer(i)==c_null_char) exit
      message(i:i)=buffer(i)
    end do
  end subroutine

  subroutine pack_layers(layers,thicknesses,p,status,message)
    type(physical_parameters), intent(in) :: layers(:)
    real(c_double), intent(in) :: thicknesses(:)
    real(c_double), allocatable, intent(out) :: p(:,:)
    integer, intent(out) :: status
    character(*), intent(out) :: message
    integer :: i, alloc_status
    status=DISPER26_INVALID_ARGUMENT
    message='layers and thicknesses must have equal lengths'
    if (size(layers)/=size(thicknesses)) return
    allocate(p(6,size(layers)),stat=alloc_status)
    if (alloc_status/=0) then
      status=DISPER26_CALCULATION_FAILED
      message='Cannot allocate layer buffer'
      return
    end if
    do i=1,size(layers)
      p(:,i)=pack_parameter(layers(i))
    end do
    status=DISPER26_OK
    message=''
  end subroutine

  subroutine search_roots(layers,thicknesses,half_space,omega,c_min,c_max,n_search,wave_type,roots,status,message)
    type(physical_parameters), intent(in) :: layers(:),half_space
    real(c_double), intent(in) :: thicknesses(:),omega,c_min,c_max
    integer, intent(in) :: n_search
    character(*), intent(in) :: wave_type
    real(c_double), allocatable, intent(out) :: roots(:)
    integer, intent(out) :: status
    character(*), intent(out) :: message
    real(c_double), allocatable :: p(:,:),buffer(:)
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: count,capacity
    integer :: alloc_status
    allocate(roots(0))
    call pack_layers(layers,thicknesses,p,status,message)
    if (status/=DISPER26_OK) return
    if (n_search<=0 .or. n_search>=huge(0_c_int)-1) then
      status=DISPER26_INVALID_ARGUMENT
      message='n_search must be positive and fit in a C integer with room for the root buffer'
      return
    end if
    capacity=int(n_search,c_int)+1_c_int
    allocate(buffer(capacity),stat=alloc_status)
    if (alloc_status/=0) then
      status=DISPER26_CALCULATION_FAILED
      message='Cannot allocate root buffer'
      return
    end if
    status=c_roots(size(layers,kind=c_int),p,thicknesses,pack_parameter(half_space),omega,c_min,c_max, &
        int(n_search,c_int),wave_code(wave_type),buffer,capacity,count,msg,message_capacity)
    ! Retry only if floating-point scan accumulation produced more roots than expected.
    if (status==DISPER26_BUFFER_TOO_SMALL) then
      deallocate(buffer)
      capacity=count
      allocate(buffer(capacity),stat=alloc_status)
      if (alloc_status/=0) then
        status=DISPER26_CALCULATION_FAILED
        message='Cannot allocate root buffer'
        return
      end if
      status=c_roots(size(layers,kind=c_int),p,thicknesses,pack_parameter(half_space),omega,c_min,c_max, &
          int(n_search,c_int),wave_code(wave_type),buffer,capacity,count,msg,message_capacity)
    end if
    call copy_message(msg,message)
    if (status==DISPER26_OK) then
      deallocate(roots)
      allocate(roots(count),stat=alloc_status)
      if (alloc_status/=0) then
        status=DISPER26_CALCULATION_FAILED
        message='Cannot allocate returned roots'
        return
      end if
      roots=buffer(:count)
    end if
  end subroutine

  subroutine compute_group_velocity_ad(layers,thicknesses,half_space,omega,c_phase,wave_type,value,status,message)
    type(physical_parameters), intent(in) :: layers(:),half_space
    real(c_double), intent(in) :: thicknesses(:),omega,c_phase
    character(*), intent(in) :: wave_type
    real(c_double), intent(out) :: value
    integer, intent(out) :: status
    character(*), intent(out) :: message
    real(c_double), allocatable :: p(:,:)
    character(kind=c_char) :: msg(message_capacity)
    value=ieee_value(0.0_c_double,ieee_quiet_nan)
    call pack_layers(layers,thicknesses,p,status,message)
    if (status/=DISPER26_OK) return
    status=c_group_velocity_ad(size(layers,kind=c_int),p,thicknesses,pack_parameter(half_space), &
        omega,c_phase,wave_code(wave_type),value,msg,message_capacity)
    call copy_message(msg,message)
  end subroutine

  subroutine compute_hv_ratio(layers,thicknesses,half_space,omega,c_phase,wave_type,value,status,message)
    type(physical_parameters), intent(in) :: layers(:),half_space
    real(c_double), intent(in) :: thicknesses(:),omega,c_phase
    character(*), intent(in) :: wave_type
    real(c_double), intent(out) :: value
    integer, intent(out) :: status
    character(*), intent(out) :: message
    real(c_double), allocatable :: p(:,:)
    character(kind=c_char) :: msg(message_capacity)
    value=ieee_value(0.0_c_double,ieee_quiet_nan)
    call pack_layers(layers,thicknesses,p,status,message)
    if (status/=DISPER26_OK) return
    status=c_hv_ratio(size(layers,kind=c_int),p,thicknesses,pack_parameter(half_space), &
        omega,c_phase,wave_code(wave_type),value,msg,message_capacity)
    call copy_message(msg,message)
  end subroutine

  subroutine compute_dispersion_romberg_velocity_iso(depths,rhos,vps,vss,is_discontinuity, &
      period,wave_type,c,u,hv,level,relative_error,status,message,c_guess,max_steps,tolerance,phase_only, &
      adaptive_root_scan,initial_root_window_frac,extrapolate_start_level)
    real(c_double), intent(in) :: depths(:),rhos(:),vps(:),vss(:),period
    logical, intent(in) :: is_discontinuity(:)
    character(*), intent(in) :: wave_type
    real(c_double), intent(out) :: c,u,hv,relative_error
    integer, intent(out) :: level,status
    character(*), intent(out) :: message
    real(c_double), optional, intent(in) :: c_guess,tolerance,initial_root_window_frac
    integer, optional, intent(in) :: max_steps,extrapolate_start_level
    logical, optional, intent(in) :: phase_only,adaptive_root_scan
    real(c_double), allocatable :: p(:,:)
    integer(c_int), allocatable :: flags(:)
    real(c_double) :: guess,tol,window
    integer(c_int) :: steps,po,adaptive,start,c_level
    integer :: n,alloc_status
    character(kind=c_char) :: msg(message_capacity)
    c=ieee_value(0.0_c_double,ieee_quiet_nan)
    u=c
    hv=c
    relative_error=c
    level=-1
    status=DISPER26_INVALID_ARGUMENT
    message='All GRID arrays must have the same nonzero length'
    n=size(depths)
    if (n==0 .or. size(is_discontinuity)/=n) return
    if (size(rhos)/=n) return
    if (size(vps)/=n) return
    if (size(vss)/=n) return
    allocate(p(3,n),flags(n),stat=alloc_status)
    if (alloc_status/=0) then
      status=DISPER26_CALCULATION_FAILED
      message='Cannot allocate GRID buffers'
      return
    end if
    p(1,:)=rhos
    p(2,:)=vps
    p(3,:)=vss
    flags=merge(1_c_int,0_c_int,is_discontinuity)
    guess=3.0_c_double
    tol=1e-10_c_double
    steps=5
    po=0
    adaptive=1
    window=0.0_c_double
    start=0
    if (present(c_guess)) guess=c_guess
    if (present(tolerance)) tol=tolerance
    if (present(max_steps)) steps=int(max_steps,c_int)
    if (present(phase_only)) po=merge(1_c_int,0_c_int,phase_only)
    if (present(adaptive_root_scan)) adaptive=merge(1_c_int,0_c_int,adaptive_root_scan)
    if (present(initial_root_window_frac)) window=initial_root_window_frac
    if (present(extrapolate_start_level)) start=int(extrapolate_start_level,c_int)
    status=c_grid_iso(int(n,c_int),depths,p,flags,period,wave_code(wave_type),guess,steps,tol,po, &
        adaptive,window,start,c,u,hv,c_level,relative_error,msg,message_capacity)
    level=int(c_level)
    call copy_message(msg,message)
  end subroutine

  subroutine compute_dispersion_romberg_velocity_vti(depths,rhos,vphs,vpvs,vshs,vsvs,etas,is_discontinuity, &
      period,wave_type,c,u,hv,level,relative_error,status,message,c_guess,max_steps,tolerance,phase_only, &
      adaptive_root_scan,initial_root_window_frac,extrapolate_start_level)
    real(c_double), intent(in) :: depths(:),rhos(:),vphs(:),vpvs(:),vshs(:),vsvs(:),etas(:),period
    logical, intent(in) :: is_discontinuity(:)
    character(*), intent(in) :: wave_type
    real(c_double), intent(out) :: c,u,hv,relative_error
    integer, intent(out) :: level,status
    character(*), intent(out) :: message
    real(c_double), optional, intent(in) :: c_guess,tolerance,initial_root_window_frac
    integer, optional, intent(in) :: max_steps,extrapolate_start_level
    logical, optional, intent(in) :: phase_only,adaptive_root_scan
    real(c_double), allocatable :: p(:,:)
    integer(c_int), allocatable :: flags(:)
    real(c_double) :: guess,tol,window
    integer(c_int) :: steps,po,adaptive,start,c_level
    integer :: n,alloc_status
    character(kind=c_char) :: msg(message_capacity)
    c=ieee_value(0.0_c_double,ieee_quiet_nan)
    u=c
    hv=c
    relative_error=c
    level=-1
    status=DISPER26_INVALID_ARGUMENT
    message='All GRID arrays must have the same nonzero length'
    n=size(depths)
    if (n==0 .or. size(is_discontinuity)/=n) return
    if (size(rhos)/=n) return
    if (size(vphs)/=n) return
    if (size(vpvs)/=n) return
    if (size(vshs)/=n) return
    if (size(vsvs)/=n) return
    if (size(etas)/=n) return
    allocate(p(6,n),flags(n),stat=alloc_status)
    if (alloc_status/=0) then
      status=DISPER26_CALCULATION_FAILED
      message='Cannot allocate GRID buffers'
      return
    end if
    p(1,:)=rhos
    p(2,:)=vphs
    p(3,:)=vpvs
    p(4,:)=vshs
    p(5,:)=vsvs
    p(6,:)=etas
    flags=merge(1_c_int,0_c_int,is_discontinuity)
    guess=3.0_c_double
    tol=1e-10_c_double
    steps=5
    po=0
    adaptive=1
    window=0.0_c_double
    start=0
    if (present(c_guess)) guess=c_guess
    if (present(tolerance)) tol=tolerance
    if (present(max_steps)) steps=int(max_steps,c_int)
    if (present(phase_only)) po=merge(1_c_int,0_c_int,phase_only)
    if (present(adaptive_root_scan)) adaptive=merge(1_c_int,0_c_int,adaptive_root_scan)
    if (present(initial_root_window_frac)) window=initial_root_window_frac
    if (present(extrapolate_start_level)) start=int(extrapolate_start_level,c_int)
    status=c_grid_vti(int(n,c_int),depths,p,flags,period,wave_code(wave_type),guess,steps,tol,po, &
        adaptive,window,start,c,u,hv,c_level,relative_error,msg,message_capacity)
    level=int(c_level)
    call copy_message(msg,message)
  end subroutine
end module disper26
